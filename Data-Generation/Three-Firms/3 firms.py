import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize


DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")
#update parameter for how many steps per update
UPDATE_EVERY = 4
N_FIRMS = 3

#environment parameters
C     = 1.0       #marginal cost (symmetric)
A     = 2.0       #product quality
A0    = 0.0       #outside option
MU    = 0.25      #horizontal differentiation
DELTA = 0.95      #discount factor

#training configuration
N_SESSIONS    = 5 #fewer sessions than baseline 
T_MAX         = 2_000_000 #maximum number of steps
EVAL_WINDOW   = 100_000 #evaluation window
BATCH_SIZE    = 64
MIN_BUFFER    = 1_000
SEED_BASE     = 42

#SAC hyperparameters
LR             = 3e-4      #learning rate for all networks
TAU            = 0.005     #soft update rate
HIDDEN         = 64        #hidden layer size
TARGET_ENTROPY = -1.0      #target entropy is -dim(action)
INIT_ALPHA     = 0.2       #initial entropy temperature
GRAD_CLIP      = 1.0       #gradient norm clipping
ALPHA_FREEZE_STEP = 100_000 #freezing entropy tuning after this step

#convergence checking
CONV_CHECK_EVERY = 10_000  #evaluate reference grid every N steps
CONV_N_CHECKS    = 5       
CONV_TOL         = 0.03    
#with 3 firms the reference grid is N_REF^3; use 8 to keep it at 512 states
N_REF            = 8

#learning rate decay schedule
LR_DECAY_START = 500_000   
LR_DECAY_END   = 1_800_000 
LR_MIN         = 1e-6      

#setting up the environment
def demand_i(pi, prices_others):
    """Logit demand for firm i given its price and a list/array of other prices."""
    num = np.exp((A - pi) / MU)
    denom = num + np.exp(A0 / MU)
    for pj in prices_others:
        denom += np.exp((A - pj) / MU)
    return num / denom

#profit function for firm i
def profit_i(pi, prices_others):
    return (pi - C) * demand_i(pi, prices_others)


def neg_joint(pvec):
    total = 0.0
    for i in range(N_FIRMS):
        others = list(pvec[:i]) + list(pvec[i+1:])
        total -= profit_i(pvec[i], others)
    return total
#nash equilibrium via iterated BR and monopoly via joint profit maximisation


def compute_benchmarks():
    #initial guess
    p = 1.5
    for _ in range(1000):
        others = [p] * (N_FIRMS - 1)
        #find profit-maximising price given all opponents charge p
        res = minimize_scalar(lambda pi: -profit_i(pi, others),
                              bounds=(C, C + 3), method="bounded")
        if abs(res.x - p) < 1e-10:
            break
        p = res.x
    p_nash = p
    pi_nash = profit_i(p_nash, [p_nash] * (N_FIRMS - 1))

    x0 = [1.8] * N_FIRMS
    res = minimize(neg_joint, x0, bounds=[(C, C + 3)] * N_FIRMS)
    p_mono = res.x[0]
    pi_mono = -neg_joint(res.x) / N_FIRMS
    return p_nash, p_mono, pi_nash, pi_mono

#computing benchmarks
P_NASH, P_MONO, PI_NASH, PI_MONO = compute_benchmarks()
SPREAD = P_MONO - P_NASH
P_MIN  = P_NASH - 0.3 * SPREAD
P_MAX  = P_MONO + 0.3 * SPREAD
CONV_PRICE_TOL = CONV_TOL * (P_MAX - P_MIN)


#normalising the price
def normalise_price(p):
    #mapping price to [0,1]
    return (p - P_MIN) / (P_MAX - P_MIN)

#mapping tanh output [-1,1] to [p_min,p_max]
def unnormalise_action(a):
    return P_MIN + (a + 1) * 0.5 * (P_MAX - P_MIN)

#calculating collusion Delta
def collusion_index(avg_profit):
    return (avg_profit - PI_NASH) / (PI_MONO - PI_NASH)


#building 8x8x8 = 512 reference states for convergence checking
#each state is (normalised own price, normalised opp1 price, normalised opp2 price)
_ref_prices = np.linspace(P_MIN, P_MAX, N_REF)
_ref_states_list = []
for p_own in _ref_prices:
    for p_opp1 in _ref_prices:
        for p_opp2 in _ref_prices:
            _ref_states_list.append([normalise_price(p_own),
                                     normalise_price(p_opp1),
                                     normalise_price(p_opp2)])
            


REF_STATES = torch.tensor(_ref_states_list, dtype=torch.float32, device=DEVICE)
N_REF_STATES = len(_ref_states_list)


STATE_DIM = N_FIRMS  #each agent sees (own_price, opp1, opp2)

#defining the xavier uniform for initialisation
def xavier_uniform(out_dim, in_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return (torch.rand(out_dim, in_dim, device=DEVICE) * 2 * limit - limit)

#small uniform initialisation for output layers (±0.003)
def small_uniform(out_dim, in_dim, bound=3e-3):
    return (torch.rand(out_dim, in_dim, device=DEVICE) * 2 * bound - bound)

#actor network: state -> (mean, log_std) for Gaussian policy
#architecture: 3 -> 64 -> 64 -> 2 (mu and log_sigma outputs)
#input is 3 because each agent observes own price + 2 opponent prices
def make_actor(state_dim=STATE_DIM, hidden=HIDDEN):
    params = {
        "l1_w": xavier_uniform(hidden, state_dim),
        "l1_b": torch.zeros(hidden, device=DEVICE),
        "l2_w": xavier_uniform(hidden, hidden),
        "l2_b": torch.zeros(hidden, device=DEVICE),
        "mu_w": small_uniform(1, hidden),
        "mu_b": torch.zeros(1, device=DEVICE),
        "ls_w": small_uniform(1, hidden),
        "ls_b": torch.zeros(1, device=DEVICE),
    }
    for v in params.values():
        v.requires_grad_(True)
    return params

#critic network: (state, action) -> Q-value
#architecture: 4 -> 64 -> 64 -> 1 (input is 3 state dims + 1 action dim)
def make_critic(state_dim=STATE_DIM, action_dim=1, hidden=HIDDEN):
    params = {
        "l1_w": xavier_uniform(hidden, state_dim + action_dim),
        "l1_b": torch.zeros(hidden, device=DEVICE),
        "l2_w": xavier_uniform(hidden, hidden),
        "l2_b": torch.zeros(hidden, device=DEVICE),
        "l3_w": small_uniform(1, hidden),
        "l3_b": torch.zeros(1, device=DEVICE),
    }
    for v in params.values():
        v.requires_grad_(True)
    return params

#forward pass through the actor network, takes normalised state and outputs mu and log std
def actor_forward(params, state):
    h = F.relu(state @ params["l1_w"].T + params["l1_b"])
    h = F.relu(h @ params["l2_w"].T + params["l2_b"])
    mu = h @ params["mu_w"].T + params["mu_b"]
    log_std = (h @ params["ls_w"].T + params["ls_b"]).clamp(-20, 2)
    return mu, log_std

#sample an action from the current policy and compute its log-probability
def actor_sample(params, state):
    mu, log_std = actor_forward(params, state)
    dist = Normal(mu, log_std.exp())
    u = dist.rsample()
    action = torch.tanh(u)
    log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
    return action.squeeze(-1), log_prob.squeeze(-1)

#greedy (deterministic) action taking the mean, no sampling noise
def actor_deterministic(params, state):
    mu, _ = actor_forward(params, state)
    return torch.tanh(mu).squeeze(-1)

#forward pass through a critic network takes (state, action) and outputs a scalar Q-value estimate
def critic_forward(params, state, action):
    if action.dim() == 1:
        action = action.unsqueeze(-1)
    x = torch.cat([state, action], dim=-1)
    h = F.relu(x @ params["l1_w"].T + params["l1_b"])
    h = F.relu(h @ params["l2_w"].T + params["l2_b"])
    return (h @ params["l3_w"].T + params["l3_b"]).squeeze(-1)

#deep copy of network parameters for target networks
def copy_params(src):
    return {k: v.clone().detach().requires_grad_(False) for k, v in src.items()}

#Polyak averaging: target = (1 - τ) * target + τ * source, slow update target networks
def soft_update(target, source, tau=TAU):
    for k in target:
        target[k].data.mul_(1 - tau).add_(tau * source[k].data)


#convergence, get each agent's greedy (deterministic) price at all 512 reference states
def get_greedy_prices_at_ref(agent):
    with torch.no_grad():
        raw = actor_deterministic(agent["actor"], REF_STATES)
        prices = P_MIN + (raw + 1) * 0.5 * (P_MAX - P_MIN)
    return prices.cpu().numpy()

#convergence check where all agents' greedy prices must be stable at all ref states
def check_convergence(ref_histories):
    for hist in ref_histories:
        if len(hist) < CONV_N_CHECKS:
            return False
        recent = np.array(hist[-CONV_N_CHECKS:])
        ranges = recent.max(axis=0) - recent.min(axis=0)
        if ranges.max() > CONV_PRICE_TOL:
            return False
    return True


#replay buffer
def make_buffer(capacity=100_000):
    return deque(maxlen=capacity)

#appending to buffer
def buffer_push(buf, state, action, reward, next_state):
    buf.append((state, action, reward, next_state))

#sampling from buffer
def buffer_sample(buf, batch_size=BATCH_SIZE):
    batch = random.sample(buf, batch_size)
    s, a, r, s2 = zip(*batch)
    return (torch.tensor(np.array(s),  dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(a),  dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(r),  dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(s2), dtype=torch.float32, device=DEVICE))


#creating the SAC agent
def make_agent(lr=LR):
    actor  = make_actor()   #policy network: state -> (mu, log_std)
    q1     = make_critic()
    q2     = make_critic()
    q1_tgt = copy_params(q1)
    q2_tgt = copy_params(q2)
    log_alpha = torch.tensor(np.log(INIT_ALPHA), device=DEVICE, requires_grad=True)

    actor_opt = torch.optim.Adam(actor.values(), lr=lr)
    q1_opt    = torch.optim.Adam(q1.values(), lr=lr)
    q2_opt    = torch.optim.Adam(q2.values(), lr=lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)

    return dict(actor=actor, q1=q1, q2=q2, q1_tgt=q1_tgt, q2_tgt=q2_tgt,
                log_alpha=log_alpha, actor_opt=actor_opt,
                q1_opt=q1_opt, q2_opt=q2_opt, alpha_opt=alpha_opt,
                buffer=make_buffer())

#choose an action given the current state
def select_action(agent, state, deterministic=False):
    s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            return actor_deterministic(agent["actor"], s).item()
        else:
            a, _ = actor_sample(agent["actor"], s)
            return a.item()

#clip gradients, applied to actor and critic parameters after each backward
def clip_grads(params, max_norm=GRAD_CLIP):
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        torch.nn.utils.clip_grad_norm_(grads, max_norm)

#SAC update, updates critics, actor, entropy temperature and target networks
def update(agent, batch_size=BATCH_SIZE, gamma=DELTA, step=0):
    if len(agent["buffer"]) < batch_size:
        return
    #sample a random batch of transitions from replay buffer
    states, actions, rewards, next_states = buffer_sample(agent["buffer"], batch_size)
    alpha = agent["log_alpha"].exp().detach()
    
    #critic update
    #bellman target: r + γ * (min Q_target(s', a') - α * log pi(a'|s'))
    with torch.no_grad():
        next_a, next_lp = actor_sample(agent["actor"], next_states)
        q1_next = critic_forward(agent["q1_tgt"], next_states, next_a)
        q2_next = critic_forward(agent["q2_tgt"], next_states, next_a)
        #min of two targets reduces overestimation
        target  = rewards + gamma * (torch.min(q1_next, q2_next) - alpha * next_lp)
        
        #update both critics by minimising MSE against the shared Bellman target
    for qp, opt in [(agent["q1"], agent["q1_opt"]), (agent["q2"], agent["q2_opt"])]:
        loss = F.mse_loss(critic_forward(qp, states, actions), target)
        opt.zero_grad()   #clears gradients from previous update
        loss.backward()   #backprop
        clip_grads(qp.values())  #clipping gradients
        opt.step()  #adam uses clipped gradients
        
    #actor update
    #maximise expected Q-value while maintaining entropy
    new_a, lp = actor_sample(agent["actor"], states)
    q_val = torch.min(critic_forward(agent["q1"], states, new_a),
                      critic_forward(agent["q2"], states, new_a))
    actor_loss = (alpha * lp - q_val).mean()
    agent["actor_opt"].zero_grad()
    actor_loss.backward()
    clip_grads(agent["actor"].values())
    agent["actor_opt"].step()
    
    #alpha update (freeze after exploration phase)
    if step < ALPHA_FREEZE_STEP:
        #when entropy is above target, α decreases to allow exploitation
        alpha_loss = -(agent["log_alpha"].exp() * (lp.detach() + TARGET_ENTROPY)).mean()
        agent["alpha_opt"].zero_grad()
        alpha_loss.backward()
        agent["alpha_opt"].step()
    
    #soft-update target networks
    soft_update(agent["q1_tgt"], agent["q1"])
    soft_update(agent["q2_tgt"], agent["q2"])


#construct state for agent i: [own_price, opp1_price, opp2_price] normalised
def make_state(prices, agent_idx):
    """State for agent_idx: [own_price, opp1_price, opp2_price] normalised."""
    own = normalise_price(prices[agent_idx])
    others = [normalise_price(prices[j]) for j in range(N_FIRMS) if j != agent_idx]
    return np.array([own] + others)


#TRAINING LOOP
def run_session(session_id):
    #new seed for each session
    seed = SEED_BASE + session_id
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    #initialise 3 independent agents (no shared weights or communication)
    agents = [make_agent() for _ in range(N_FIRMS)]
    
    #random initial prices within the allowed range
    prices_now = [np.random.uniform(P_MIN, P_MAX) for _ in range(N_FIRMS)]

    price_history = []
    profit_history = []
    converged_at = None
    
    
    #store snapshots of each agent's greedy prices at the 512 reference states
    ref_histories = [[] for _ in range(N_FIRMS)]
    
    
    #logging arrays for post-training analysis
    t0 = time.time()
    alpha_log = []
    greedy_price_log = []
    delta_log = []

    for t in range(T_MAX):
        #each agent sees (own price, opp1 price, opp2 price) from previous period, normalised to [0,1]
        states = [make_state(prices_now, i) for i in range(N_FIRMS)]

        #sample actions from stochastic policies
        raw_actions = [select_action(agents[i], states[i]) for i in range(N_FIRMS)]
        new_prices = [unnormalise_action(a) for a in raw_actions]

        #compute profits from the logit demand model for each firm
        pi = []
        for i in range(N_FIRMS):
            others = [new_prices[j] for j in range(N_FIRMS) if j != i]
            pi.append(profit_i(new_prices[i], others))
        #normalise rewards by Nash profit so r=1 means Nash, r>1 means supra-competitive
        rewards = [p / PI_NASH for p in pi]

        #construct next states (prices just chosen become next period's observed state)
        next_states = [make_state(new_prices, i) for i in range(N_FIRMS)]

        #store (s, a, r, s') transitions in each agent's replay buffer
        for i in range(N_FIRMS):
            buffer_push(agents[i]["buffer"], states[i], raw_actions[i],
                        rewards[i], next_states[i])

        #update networks every UPDATE_EVERY steps, once buffer has enough transitions
        if len(agents[0]["buffer"]) >= MIN_BUFFER and (t + 1) % UPDATE_EVERY == 0:
            for ag in agents:
                update(ag, BATCH_SIZE, step=t)
                
            #freeze entropy temperature at ALPHA_FREEZE_STEP and sync across all 3 agents
            if t + 1 == ALPHA_FREEZE_STEP:
                with torch.no_grad():
                    mean_la = np.mean([ag["log_alpha"].item() for ag in agents])
                    for ag in agents:
                        ag["log_alpha"].fill_(mean_la)
                print(f" Alpha frozen at {np.exp(mean_la):.4f}")
            
            #linear learning rate decay (analogous to epsilon decay in Q-learning)
            if t >= LR_DECAY_START:
                frac = min((t - LR_DECAY_START) / (LR_DECAY_END - LR_DECAY_START), 1.0)
                new_lr = LR * (1 - frac) + LR_MIN * frac
                for ag in agents:
                    for opt in [ag["actor_opt"], ag["q1_opt"], ag["q2_opt"]]:
                        for pg in opt.param_groups:
                            pg["lr"] = new_lr
                            
        #update current prices for the next period
        prices_now = new_prices
        price_history.append(tuple(new_prices))
        profit_history.append(tuple(pi))

        #every CONV_CHECK_EVERY steps = log + snapshot + convergence
        if (t + 1) % CONV_CHECK_EVERY == 0:
            elapsed = time.time() - t0
            
            #compute rolling collusion index from recent profits
            n_recent = min(EVAL_WINDOW, len(profit_history))
            recent = profit_history[-n_recent:]
            avg_pi = np.mean([sum(x) for x in recent]) / N_FIRMS
            d = collusion_index(avg_pi)
            
            #current entropy temperature values for logging
            alphas = [ag["log_alpha"].exp().item() for ag in agents]
            
            #get deterministic (greedy) prices at the current state for display
            with torch.no_grad():
                gps = []
                for i in range(N_FIRMS):
                    s_t = torch.tensor(states[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    gps.append(unnormalise_action(actor_deterministic(agents[i]["actor"], s_t).item()))

            #append to logs for post-training plots            
            alpha_log.append((t + 1, *alphas))
            delta_log.append((t + 1, d))
            greedy_price_log.append((t + 1, *gps))
            
            #snapshot greedy prices at ALL 512 reference states for convergence tracking
            for i in range(N_FIRMS):
                ref_histories[i].append(get_greedy_prices_at_ref(agents[i]))
            
            #print training progress
            frozen_msg = "" if t < ALPHA_FREEZE_STEP else " [α frozen]"
            gp_str = ",".join(f"{g:.4f}" for g in gps)
            alpha_str = " ".join(f"α{i+1}={a:.3f}" for i, a in enumerate(alphas))
            print(f"  S{session_id} | t={t+1:>7d} | Δ={d:.3f} "
                  f"| {alpha_str} "
                  f"| gp=({gp_str}) | {elapsed:.0f}s{frozen_msg}")
            
            #check convergence across all 3 agents
            if converged_at is None:
                if check_convergence(ref_histories):
                    converged_at = t + 1
                    print(f"  Converged at step {converged_at} "
                          f"(policy stable at all {N_REF_STATES} ref states "
                          f"for {CONV_N_CHECKS * CONV_CHECK_EVERY // 1000}k steps)")
                    break
                
                #convergence diagnostics
            if len(ref_histories[0]) >= 2:
                diag_parts = []
                for i in range(N_FIRMS):
                    diff = np.abs(ref_histories[i][-1] - ref_histories[i][-2])
                    diag_parts.append(
                        f"ag{i+1} max_Δp={diff.max():.4f} "
                        f"states>{CONV_PRICE_TOL:.4f}: "
                        f"{(diff > CONV_PRICE_TOL).sum()}/{N_REF_STATES}")
                print(f"         ref grid | {' | '.join(diag_parts)}")

   #final metrics computed over the last EVAL_WINDOW steps
    prices_arr  = np.array(price_history)
    profits_arr = np.array(profit_history)
    n_eval = min(EVAL_WINDOW, len(profits_arr))
    avg_profit = profits_arr[-n_eval:].mean(axis=0) #per-firm average profi
    avg_price  = prices_arr[-n_eval:].mean(axis=0) #per-firm average price
    d = collusion_index(avg_profit.mean()) #final collusion index
    
    #return all results and trained agents for post-training analysis
    return dict(delta=d, avg_price=avg_price, avg_profit=avg_profit,
                price_history=prices_arr, profit_history=profits_arr,
                converged_at=converged_at,
                agents=agents,
                alpha_log=np.array(alpha_log),
                delta_log=np.array(delta_log),
                greedy_price_log=np.array(greedy_price_log))


#PUNISHMENT TEST
def punishment_test(agents, price_history, deviator=0, n_steps=30):
    #pre-deviation prices the last converged prices from training
    p_star = list(price_history[-1])
    
    #compute deviator's static best response given other firms' prices
    others_star = [p_star[j] for j in range(N_FIRMS) if j != deviator]
    res = minimize_scalar(lambda p: -profit_i(p, others_star),
                          bounds=(P_MIN, P_MAX), method="bounded")
    p_dev = res.x

    prices_now = list(p_star)
    traj = [tuple(prices_now)] #record trajectory starting from pre-deviation state

    for step in range(n_steps):
        new_prices = []
        for i in range(N_FIRMS):
            if step == 0 and i == deviator:
                new_prices.append(p_dev)
            else:
                s = make_state(prices_now, i)
                new_prices.append(unnormalise_action(
                    select_action(agents[i], s, deterministic=True)))
        prices_now = new_prices
        traj.append(tuple(prices_now))
        #returns (n_steps+1, 3) array pre-deviation state + n_steps of response
    return np.array(traj)


#plotting
def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    deltas = [r["delta"] for r in results]

    #histogram of collusion index across all sessions
    axes[0, 0].hist(deltas, bins=25, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].axvline(0, color="green", ls="--", label="Nash (Δ=0)")
    axes[0, 0].axvline(1, color="red",   ls="--", label="Monopoly (Δ=1)")
    axes[0, 0].set_xlabel("Collusion Index Δ"); axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Collusion Index (mean={np.mean(deltas):.3f})")
    axes[0, 0].legend()

    #price trajectory for session 0 (last 5000 steps), now with 3 firms
    ph = results[0]["price_history"][-5000:]
    for i in range(N_FIRMS):
        axes[0, 1].plot(ph[:, i], alpha=0.7, lw=0.5, label=f"Firm {i+1}")
    axes[0, 1].axhline(P_NASH, color="green", ls="--", label=f"Nash ({P_NASH:.3f})")
    axes[0, 1].axhline(P_MONO, color="red",   ls="--", label=f"Mono ({P_MONO:.3f})")
    axes[0, 1].set_xlabel("Step (last 5000)"); axes[0, 1].set_ylabel("Price")
    axes[0, 1].set_title("Price Trajectory (Session 0)")
    axes[0, 1].legend(fontsize=7)

    #smoothed learning curve averaged across all sessions
    #average price across all 3 firms at each step, then average across sessions
    max_len = max(r["price_history"].shape[0] for r in results)
    mean_prices = np.full(max_len, np.nan)
    counts = np.zeros(max_len)
    for r in results:
        ph_mean = r["price_history"].mean(axis=1)
        n = len(ph_mean)
        mean_prices[:n] = np.nansum([mean_prices[:n], ph_mean], axis=0)
        counts[:n] += 1
    mean_prices = mean_prices / np.maximum(counts, 1)
    #smooth with a rolling window to reduce noise
    w = min(5000, max_len // 2)
    if w > 0:
        smoothed = np.convolve(mean_prices, np.ones(w) / w, mode="valid")
        axes[1, 0].plot(smoothed, color="steelblue", lw=1)
    axes[1, 0].axhline(P_NASH, color="green", ls="--", label="Nash")
    axes[1, 0].axhline(P_MONO, color="red",   ls="--", label="Monopoly")
    axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("Avg Price")
    axes[1, 0].set_title("Learning Curve"); axes[1, 0].legend()

    #punishment test on the first session with delta > 0.3
    collusive = [r for r in results if r["delta"] > 0.3]
    if collusive:
        traj = punishment_test(collusive[0]["agents"], collusive[0]["price_history"])
        steps = np.arange(-1, len(traj) - 1)
        axes[1, 1].plot(steps, traj[:, 0], "o-", ms=4, label="Firm 1 (deviator)")
        for i in range(1, N_FIRMS):
            axes[1, 1].plot(steps, traj[:, i], "s-", ms=4, label=f"Firm {i+1}")
        axes[1, 1].axhline(P_NASH, color="green", ls="--")
        axes[1, 1].axhline(P_MONO, color="red",   ls="--")
        axes[1, 1].axvline(0, color="gray", ls=":", label="Deviation")
        axes[1, 1].set_xlabel("Period"); axes[1, 1].set_ylabel("Price")
        axes[1, 1].set_title("Punishment Response"); axes[1, 1].legend(fontsize=7)
    else:
        axes[1, 1].text(0.5, 0.5, "No collusive sessions (Δ > 0.3)", ha="center", va="center")

    plt.tight_layout()
    plt.savefig("sac_results.png", dpi=200)
    print("Saved sac_results.png")


#MAIN
if __name__ == "__main__":
    #printing initial
    print(f"N_FIRMS = {N_FIRMS}")
    print(f"p_nash  = {P_NASH:.4f}   p_mono  = {P_MONO:.4f}")
    print(f"π_nash  = {PI_NASH:.4f}   π_mono  = {PI_MONO:.4f}")
    print(f"p_range = [{P_MIN:.4f}, {P_MAX:.4f}]")
    print(f"conv_tol = {CONV_PRICE_TOL:.5f} (price units)")
    print(f"ref grid = {N_REF}^{N_FIRMS} = {N_REF_STATES} states per agent")
    print(f"conv: policy stable for "
          f"{CONV_N_CHECKS}×{CONV_CHECK_EVERY//1000}k = "
          f"{CONV_N_CHECKS * CONV_CHECK_EVERY // 1000}k steps")
    print(f"alpha_freeze = step {ALPHA_FREEZE_STEP}")
    print(f"Running {N_SESSIONS} sessions × {T_MAX} steps max\n")

    total_start = time.time()
    results = []
    #printing session diagnostics
    for i in range(N_SESSIONS):
        print(f"── Session {i} ──")
        session_start = time.time()
        r = run_session(i)
        session_time = time.time() - session_start

        conv_msg = f"converged@{r['converged_at']}" if r["converged_at"] else "no convergence"
        price_str = ", ".join(f"{p:.4f}" for p in r["avg_price"])
        print(f"  Δ={r['delta']:.3f}  prices=({price_str})  "
              f"{conv_msg}  [{session_time:.0f}s]\n")
        results.append(r)

    total_time = time.time() - total_start
    deltas = [r["delta"] for r in results]
    conv_count = sum(1 for r in results if r["converged_at"] is not None)

    print("=" * 55)
    print(f"Mean Delta       = {np.mean(deltas):.3f} ± {np.std(deltas):.3f}")
    print(f"Median Delta    = {np.median(deltas):.3f}")
    print(f"Delta > 0.5      : {np.mean(np.array(deltas) > 0.5):.0%}")
    print(f"Delta > 0        : {np.mean(np.array(deltas) > 0):.0%}")
    print(f"Converged    : {conv_count}/{N_SESSIONS}")
    print(f"Total time   : {total_time / 60:.1f} minutes")

    #saving to CSV
    import csv
    print("Saving data...")

    for i, r in enumerate(results):
        with open(f"sac_session_{i}_prices.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["step"] + [f"price_{j+1}" for j in range(N_FIRMS)] + \
                     [f"profit_{j+1}" for j in range(N_FIRMS)]
            w.writerow(header)
            for t, (pr, pi) in enumerate(zip(r["price_history"], r["profit_history"])):
                w.writerow([t] + list(pr) + list(pi))

        with open(f"sac_session_{i}_logs.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["step", "delta"] + [f"alpha_{j+1}" for j in range(N_FIRMS)] + \
                     [f"greedy_p{j+1}" for j in range(N_FIRMS)]
            w.writerow(header)
            for dl, al, gl in zip(r["delta_log"], r["alpha_log"], r["greedy_price_log"]):
                row = [int(dl[0]), dl[1]] + list(al[1:]) + list(gl[1:])
                w.writerow(row)
                #saving .pt files for weights
        ags = r["agents"]
        save_dict = {}
        for idx, ag in enumerate(ags):
            save_dict[f"agent{idx+1}_actor"] = {k: v.cpu() for k, v in ag["actor"].items()}
            save_dict[f"agent{idx+1}_q1"]    = {k: v.cpu() for k, v in ag["q1"].items()}
            save_dict[f"agent{idx+1}_q2"]    = {k: v.cpu() for k, v in ag["q2"].items()}
            save_dict[f"agent{idx+1}_log_alpha"] = ag["log_alpha"].cpu()
        torch.save(save_dict, f"sac_agents_session_{i}.pt")
        
        #saving hyperparameters
    with open("sac_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["session", "delta"] + \
                 [f"avg_price_{j+1}" for j in range(N_FIRMS)] + \
                 [f"avg_profit_{j+1}" for j in range(N_FIRMS)] + \
                 ["converged_at", "steps_run"]
        w.writerow(header)
        for i, r in enumerate(results):
            row = [i, r["delta"]] + list(r["avg_price"]) + list(r["avg_profit"]) + \
                  [r["converged_at"] if r["converged_at"] else -1,
                   len(r["price_history"])]
            w.writerow(row)

    with open("sac_hyperparams.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in [("n_firms", N_FIRMS), ("c", C), ("a", A), ("a0", A0), ("mu", MU),
                      ("p_nash", P_NASH), ("p_mono", P_MONO),
                      ("pi_nash", PI_NASH), ("pi_mono", PI_MONO),
                      ("p_min", P_MIN), ("p_max", P_MAX),
                      ("lr", LR), ("tau", TAU), ("hidden", HIDDEN),
                      ("lr_decay_start", LR_DECAY_START),
                      ("lr_decay_end", LR_DECAY_END),
                      ("lr_min", LR_MIN),
                      ("target_entropy", TARGET_ENTROPY), ("init_alpha", INIT_ALPHA),
                      ("gamma", DELTA), ("batch_size", BATCH_SIZE),
                      ("grad_clip", GRAD_CLIP),
                      ("conv_check_every", CONV_CHECK_EVERY),
                      ("conv_n_checks", CONV_N_CHECKS),
                      ("conv_tol", CONV_TOL),
                      ("conv_price_tol", CONV_PRICE_TOL),
                      ("n_ref", N_REF),
                      ("n_ref_states", N_REF_STATES),
                      ("t_max", T_MAX),
                      ("n_sessions", N_SESSIONS), ("update_every", UPDATE_EVERY),
                      ("min_buffer", MIN_BUFFER), ("eval_window", EVAL_WINDOW),
                      ("seed_base", SEED_BASE),
                      ("alpha_freeze_step", ALPHA_FREEZE_STEP),
                      ("mean_delta", np.mean(deltas)),
                      ("std_delta", np.std(deltas)),
                      ("median_delta", np.median(deltas)),
                      ("frac_above_0", np.mean(np.array(deltas) > 0)),
                      ("frac_above_05", np.mean(np.array(deltas) > 0.5)),
                      ("convergence_rate", f"{conv_count}/{N_SESSIONS}"),
                      ("total_time_minutes", round(total_time / 60, 1))]:
            w.writerow([k, v])

    print(f"Saved: {N_SESSIONS}x prices CSV + {N_SESSIONS}x logs CSV + "
          f"agent weights + sac_summary.csv + sac_hyperparams.csv")

    plot_results(results)
