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

#environment parameters (asymmetric marginal costs)
C1    = 1.0       #marginal cost firm 1 (high-cost)
C2    = 0.75      #marginal cost firm 2 (low-cost)
A     = 2.0       #product quality
A0    = 0.0       #outside option
MU    = 0.25      #horizontal differentiation
DELTA = 0.95      #discount factor

#training configuration
N_SESSIONS    = 5          #fewer sessions than baseline due to compute
T_MAX         = 2_000_000  #maximum number of steps
EVAL_WINDOW   = 100_000    #evaluation window
BATCH_SIZE    = 64
MIN_BUFFER    = 1_000
SEED_BASE     = 42

#SAC hyperparameters
LR            = 3e-4       #learning rate for all networks
TAU           = 0.005      #soft update rate
HIDDEN        = 64         #hidden layer size
TARGET_ENTROPY = -1.0      #target entropy is -dim(action)
INIT_ALPHA    = 0.2        #initial entropy temperature
GRAD_CLIP     = 1.0        #gradient norm clipping
ALPHA_FREEZE_STEP = 100_000 #freezing entropy tuning after this step

#convergence checking
CONV_CHECK_EVERY = 10_000  #evaluate reference grid every N steps
CONV_N_CHECKS    = 5       #consecutive stable checks needed
CONV_TOL         = 0.03    #max price change as fraction of price range
N_REF            = 15      #15x15 = 225 reference states

#learning rate decay schedule
LR_DECAY_START = 500_000   #start decaying after alpha freeze
LR_DECAY_END   = 1_800_000 #LR reaches minimum by this step
LR_MIN         = 1e-6      #minimum learning rate is effectively zero


#setting up the environment

#demand function (same for both firms, only prices differ)
def demand(pi, pj):
    num = np.exp((A - pi) / MU)
    return num / (num + np.exp((A - pj) / MU) + np.exp(A0 / MU))

#profit function now takes firm-specific marginal cost ci
def profit(pi, pj, ci):
    return (pi - ci) * demand(pi, pj)

#negative total industry profit (minimised to find joint-profit-maximising prices)
#each firm uses its own marginal cost
def neg_joint(prices):
    pa, pb = prices
    return -profit(pa, pb, C1) - profit(pb, pa, C2)

#nash equilibrium via iterated BR (now asymmetric)
def compute_benchmarks():
    #initial guesses
    p1, p2 = 1.5, 1.5
    for _ in range(1000):
        #firm 1 best responds to p2 using its own cost C1
        r1 = minimize_scalar(lambda pi: -profit(pi, p2, C1), bounds=(C1, C1+3), method="bounded")
        #firm 2 best responds to p1 using its own cost C2
        r2 = minimize_scalar(lambda pi: -profit(pi, p1, C2), bounds=(C2, C2+3), method="bounded")
        #check if both best responses equal current prices (fixed point)
        if abs(r1.x - p1) < 1e-10 and abs(r2.x - p2) < 1e-10:
            break
        p1, p2 = r1.x, r2.x
    p_nash1, p_nash2 = p1, p2
    pi_nash1 = profit(p_nash1, p_nash2, C1)  #firm 1 nash profit
    pi_nash2 = profit(p_nash2, p_nash1, C2)  #firm 2 nash profit
    #monopoly outcome via joint profit maximisation
    res = minimize(neg_joint, [1.8, 1.8], bounds=[(min(C1, C2), max(C1, C2)+3)]*2)
    p_mono1, p_mono2 = res.x                 #now potentially different monopoly prices
    pi_mono1 = profit(p_mono1, p_mono2, C1)  #firm 1 monopoly profit
    pi_mono2 = profit(p_mono2, p_mono1, C2)  #firm 2 monopoly profit
    return p_nash1, p_nash2, p_mono1, p_mono2, pi_nash1, pi_nash2, pi_mono1, pi_mono2


#computing benchmarks (now 8 values instead of 4 due to asymmetry)
P_NASH1, P_NASH2, P_MONO1, P_MONO2, PI_NASH1, PI_NASH2, PI_MONO1, PI_MONO2 = compute_benchmarks()
P_NASH_AVG = (P_NASH1 + P_NASH2) / 2
P_MONO_AVG = (P_MONO1 + P_MONO2) / 2
#price range spans from lowest nash to highest monopoly price
SPREAD = max(P_MONO1, P_MONO2) - min(P_NASH1, P_NASH2)
P_MIN  = min(P_NASH1, P_NASH2) - 0.3 * SPREAD
P_MAX  = max(P_MONO1, P_MONO2) + 0.3 * SPREAD
CONV_PRICE_TOL = CONV_TOL * (P_MAX - P_MIN)   #absolute price tolerance


#normalising the price
def normalise_price(p):
    #mapping price to [0,1]
    return (p - P_MIN) / (P_MAX - P_MIN)

#mapping tanh output [-1,1] to [p_min,p_max]
def unnormalise_action(a):
    return P_MIN + (a + 1) * 0.5 * (P_MAX - P_MIN)

#calculating collusion Delta (now per-firm then averaged)
#each firm's Delta_i = (avg_pi_i - pi_nash_i) / (pi_mono_i - pi_nash_i)
def collusion_index(avg_pi1, avg_pi2):
    d1 = (avg_pi1 - PI_NASH1) / (PI_MONO1 - PI_NASH1)
    d2 = (avg_pi2 - PI_NASH2) / (PI_MONO2 - PI_NASH2)
    return (d1 + d2) / 2


#building 15x15 grid of reference states for convergence checking
#each state is a (normalised own price, normalised opponent price) pair
_ref_prices = np.linspace(P_MIN, P_MAX, N_REF)
_ref_states_list = []
for p_own in _ref_prices:
    for p_opp in _ref_prices:
        _ref_states_list.append([normalise_price(p_own), normalise_price(p_opp)])
REF_STATES = torch.tensor(_ref_states_list, dtype=torch.float32, device=DEVICE)


#defining the xavier uniform for initialisation
def xavier_uniform(out_dim, in_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return (torch.rand(out_dim, in_dim, device=DEVICE) * 2 * limit - limit)

#small uniform initialisation for output layers (+/-0.003)
#prevents extreme initial Q-values or actions deep in tanh saturation
def small_uniform(out_dim, in_dim, bound=3e-3):
    return (torch.rand(out_dim, in_dim, device=DEVICE) * 2 * bound - bound)


#actor network: state -> (mean, log_std) for Gaussian policy
#architecture: 2 -> 64 -> 64 -> 2 (mu and log_sigma outputs)
def make_actor(state_dim=2, hidden=HIDDEN):
    params = {
        "l1_w": xavier_uniform(hidden, state_dim),  #hidden layer 1 weights
        "l1_b": torch.zeros(hidden, device=DEVICE),
        "l2_w": xavier_uniform(hidden, hidden),      #hidden layer 2 weights
        "l2_b": torch.zeros(hidden, device=DEVICE),
        "mu_w": small_uniform(1, hidden),             #mean output head
        "mu_b": torch.zeros(1, device=DEVICE),
        "ls_w": small_uniform(1, hidden),             #log-std output head
        "ls_b": torch.zeros(1, device=DEVICE),
    }
    for v in params.values():
        v.requires_grad_(True)
    return params

#critic network: (state, action) -> Q-value
#architecture: 3 -> 64 -> 64 -> 1
def make_critic(state_dim=2, action_dim=1, hidden=HIDDEN):
    params = {
        "l1_w": xavier_uniform(hidden, state_dim + action_dim),  #takes state + action
        "l1_b": torch.zeros(hidden, device=DEVICE),
        "l2_w": xavier_uniform(hidden, hidden),
        "l2_b": torch.zeros(hidden, device=DEVICE),
        "l3_w": small_uniform(1, hidden),              #scalar Q-value output
        "l3_b": torch.zeros(1, device=DEVICE),
    }
    for v in params.values():
        v.requires_grad_(True)
    return params


#forward pass through the actor network, takes normalised state and outputs mu and log std
def actor_forward(params, state):
    #two hidden layers with ReLU activation
    h = F.relu(state @ params["l1_w"].T + params["l1_b"])
    h = F.relu(h @ params["l2_w"].T + params["l2_b"])
    #mean head unbounded, represents the centre of the Gaussian
    mu = h @ params["mu_w"].T + params["mu_b"]
    #log-std head: clamped to [-20, 2] to prevent numerical instability
    log_std = (h @ params["ls_w"].T + params["ls_b"]).clamp(-20, 2)
    return mu, log_std


#sample an action from the current policy and compute its log-probability
def actor_sample(params, state):
    mu, log_std = actor_forward(params, state)
    dist = Normal(mu, log_std.exp())
    #rsample uses the reparameterisation trick where u = mu + std * noise
    u = dist.rsample()
    #squash through tanh to bound action to (-1, 1), later mapped to price
    action = torch.tanh(u)
    #correct the log-probability for the tanh change of variables
    log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
    return action.squeeze(-1), log_prob.squeeze(-1)


#greedy (deterministic) action taking the mean, no sampling noise
def actor_deterministic(params, state):
    mu, _ = actor_forward(params, state)
    return torch.tanh(mu).squeeze(-1)

#forward pass through a critic network takes (state, action) and outputs a scalar Q-value estimate
def critic_forward(params, state, action):
    if action.dim() == 1:
        action = action.unsqueeze(-1)    #ensure action is (batch, 1) for concat
    x = torch.cat([state, action], dim=-1)  #combine state and action as input
    h = F.relu(x @ params["l1_w"].T + params["l1_b"])
    h = F.relu(h @ params["l2_w"].T + params["l2_b"])
    return (h @ params["l3_w"].T + params["l3_b"]).squeeze(-1)

#deep copy of network parameters for target networks
def copy_params(src):
    return {k: v.clone().detach().requires_grad_(False) for k, v in src.items()}

#Polyak averaging: target = (1 - tau) * target + tau * source, slow update target networks
def soft_update(target, source, tau=TAU):
    for k in target:
        target[k].data.mul_(1 - tau).add_(tau * source[k].data)


#convergence, get each agent's greedy (deterministic) price at all 225 reference states
def get_greedy_prices_at_ref(agent):
    with torch.no_grad():
        raw_actions = actor_deterministic(agent["actor"], REF_STATES)  #(225,) actions in (-1,1)
        prices = P_MIN + (raw_actions + 1) * 0.5 * (P_MAX - P_MIN)    #map to price interval
    return prices.cpu().numpy()

#convergence check where policy has converged if, for both agents, the greedy price at every reference state has changed by less than CONV_PRICE_TOL
def check_convergence_calvano(ref_history_1, ref_history_2):
    if len(ref_history_1) < CONV_N_CHECKS:
        return False  #not enough snapshots yet
    for hist in [ref_history_1, ref_history_2]:
        recent = np.array(hist[-CONV_N_CHECKS:])
        ranges = recent.max(axis=0) - recent.min(axis=0)  #max price swing per state
        if ranges.max() > CONV_PRICE_TOL:  #any state exceeds tolerance
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
    actor  = make_actor()       #policy network: state -> (mu, log_std)
    q1     = make_critic()      #twin critic 1: (state, action) -> Q-value
    q2     = make_critic()      #twin critic 2: (state, action) -> Q-value
    q1_tgt = copy_params(q1)    #target critic 1 (slowly updated copy for stable Bellman targets)
    q2_tgt = copy_params(q2)    #target critic 2
    #learnable entropy temperature, initialised to log(0.2)
    log_alpha = torch.tensor(np.log(INIT_ALPHA), device=DEVICE, requires_grad=True)
    #separate Adam optimiser for each component
    actor_opt = torch.optim.Adam(actor.values(), lr=lr)
    q1_opt    = torch.optim.Adam(q1.values(), lr=lr)
    q2_opt    = torch.optim.Adam(q2.values(), lr=lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)

    return dict(actor=actor, q1=q1, q2=q2, q1_tgt=q1_tgt, q2_tgt=q2_tgt,
                log_alpha=log_alpha, actor_opt=actor_opt,
                q1_opt=q1_opt, q2_opt=q2_opt, alpha_opt=alpha_opt,
                buffer=make_buffer())  #each agent gets its own replay buffer


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
        return  #not enough experience to train yet

    #sample a random batch of transitions from replay buffer
    states, actions, rewards, next_states = buffer_sample(agent["buffer"], batch_size)
    alpha = agent["log_alpha"].exp().detach()  #current entropy temperature

    #critic update
    #bellman target: r + gamma * (min Q_target(s', a') - alpha * log pi(a'|s'))
    with torch.no_grad():
        next_a, next_lp = actor_sample(agent["actor"], next_states)  #sample next action from current policy
        q1_next = critic_forward(agent["q1_tgt"], next_states, next_a)  #target critic 1
        q2_next = critic_forward(agent["q2_tgt"], next_states, next_a)
        #min of two targets reduces overestimation
        target  = rewards + gamma * (torch.min(q1_next, q2_next) - alpha * next_lp)

    #update both critics by minimising MSE against the shared Bellman target
    for qp, opt in [(agent["q1"], agent["q1_opt"]), (agent["q2"], agent["q2_opt"])]:
        loss = F.mse_loss(critic_forward(qp, states, actions), target)
        opt.zero_grad()   #clears gradients from previous update
        loss.backward()   #backprop
        clip_grads(qp.values())  #clipping gradients
        opt.step()        #adam uses clipped gradients

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
        #when entropy is above target, alpha decreases to allow exploitation
        alpha_loss = -(agent["log_alpha"].exp() * (lp.detach() + TARGET_ENTROPY)).mean()
        agent["alpha_opt"].zero_grad()
        alpha_loss.backward()
        agent["alpha_opt"].step()

    #soft-update target networks
    soft_update(agent["q1_tgt"], agent["q1"])
    soft_update(agent["q2_tgt"], agent["q2"])


#TRAINING LOOP
def run_session(session_id):
    #new seed for each session
    seed = SEED_BASE + session_id
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    #initialise two independent agents (no shared weights or communication)
    ag1 = make_agent()
    ag2 = make_agent()

    #random initial prices within the allowed range
    p1 = np.random.uniform(P_MIN, P_MAX)
    p2 = np.random.uniform(P_MIN, P_MAX)

    prices, profits = [], []
    converged_at = None

    #store snapshots of each agent's greedy prices at the 225 reference states
    ref_history_1 = []
    ref_history_2 = []

    t0 = time.time()

    #logging arrays for post-training analysis
    alpha_log = []
    greedy_price_log = []
    delta_log = []

    for t in range(T_MAX):
        #each agent sees (own price, opponent price) from previous period, normalised to [0,1]
        state1 = np.array([normalise_price(p1), normalise_price(p2)])
        state2 = np.array([normalise_price(p2), normalise_price(p1)])

        #sample actions from stochastic policies
        a1 = select_action(ag1, state1)
        a2 = select_action(ag2, state2)

        #map actions back to actual prices in [P_MIN, P_MAX]
        new_p1 = unnormalise_action(a1)
        new_p2 = unnormalise_action(a2)

        #compute profits from the logit demand model using firm-specific costs
        pi1 = profit(new_p1, new_p2, C1)  #firm 1 uses cost C1=1.0
        pi2 = profit(new_p2, new_p1, C2)  #firm 2 uses cost C2=0.75
        #normalise rewards by each firm's own Nash profit
        r1 = pi1 / PI_NASH1
        r2 = pi2 / PI_NASH2

        #construct next states (prices just chosen become next period's observed state)
        next_state1 = np.array([normalise_price(new_p1), normalise_price(new_p2)])
        next_state2 = np.array([normalise_price(new_p2), normalise_price(new_p1)])

        #store (s, a, r, s') transitions in each agent's replay buffer
        buffer_push(ag1["buffer"], state1, a1, r1, next_state1)
        buffer_push(ag2["buffer"], state2, a2, r2, next_state2)

        #update networks every UPDATE_EVERY steps, once buffer has enough transitions
        if len(ag1["buffer"]) >= MIN_BUFFER and (t + 1) % UPDATE_EVERY == 0:
            update(ag1, BATCH_SIZE, step=t)
            update(ag2, BATCH_SIZE, step=t)

            #freeze entropy temperature at ALPHA_FREEZE_STEP and sync them
            if t + 1 == ALPHA_FREEZE_STEP:
                with torch.no_grad():
                    mean_log_alpha = (ag1["log_alpha"].item() + ag2["log_alpha"].item()) / 2
                    ag1["log_alpha"].fill_(mean_log_alpha)
                    ag2["log_alpha"].fill_(mean_log_alpha)
                print(f"  Alpha frozen at {np.exp(mean_log_alpha):.4f} (mean of both agents)")

            #linear learning rate decay (analogous to epsilon decay in Q-learning)
            if t >= LR_DECAY_START:
               frac = min((t - LR_DECAY_START) / (LR_DECAY_END - LR_DECAY_START), 1.0)
               new_lr = LR * (1 - frac) + LR_MIN * frac  #linear interpolation
               for ag in [ag1, ag2]:
                   for opt in [ag["actor_opt"], ag["q1_opt"], ag["q2_opt"]]:
                       for pg in opt.param_groups:
                           pg["lr"] = new_lr

        #update current prices for the next period
        p1, p2 = new_p1, new_p2
        prices.append((p1, p2))
        profits.append((pi1, pi2))

        #every CONV_CHECK_EVERY steps = log + snapshot + convergence
        if (t + 1) % CONV_CHECK_EVERY == 0:
            elapsed = time.time() - t0

            #compute rolling collusion index Delta from recent profits (per-firm then averaged)
            n_recent = min(EVAL_WINDOW, len(profits))
            recent = profits[-n_recent:]
            avg_pi1 = np.mean([x[0] for x in recent])  #firm 1 average profit
            avg_pi2 = np.mean([x[1] for x in recent])  #firm 2 average profit
            d = collusion_index(avg_pi1, avg_pi2)

            #current entropy temperature values for logging
            a1_alpha = ag1["log_alpha"].exp().item()
            a2_alpha = ag2["log_alpha"].exp().item()

            #get deterministic (greedy) prices at the current state for display
            with torch.no_grad():
                s1_t = torch.tensor(state1, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                s2_t = torch.tensor(state2, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                gp1_now = unnormalise_action(actor_deterministic(ag1["actor"], s1_t).item())
                gp2_now = unnormalise_action(actor_deterministic(ag2["actor"], s2_t).item())

            #append to logs for post-training plots
            alpha_log.append((t + 1, a1_alpha, a2_alpha))
            delta_log.append((t + 1, d))
            greedy_price_log.append((t + 1, gp1_now, gp2_now))

            #snapshot greedy prices at ALL 225 reference states for convergence tracking
            ref_history_1.append(get_greedy_prices_at_ref(ag1))
            ref_history_2.append(get_greedy_prices_at_ref(ag2))

            #print training progress
            frozen_msg = "" if t < ALPHA_FREEZE_STEP else " [alpha frozen]"
            print(f"  S{session_id} | t={t+1:>7d} | Delta={d:.3f} "
                  f"| alpha1={a1_alpha:.3f} alpha2={a2_alpha:.3f} "
                  f"| gp=({gp1_now:.4f},{gp2_now:.4f}) | {elapsed:.0f}s{frozen_msg}")

            #check convergence
            if converged_at is None:
                if check_convergence_calvano(ref_history_1, ref_history_2):
                    converged_at = t + 1
                    print(f"  Converged at step {converged_at} "
                          f"(policy stable at all {N_REF**2} ref states "
                          f"for {CONV_N_CHECKS * CONV_CHECK_EVERY // 1000}k steps)")
                    break

            #convergence diagnostics: how far is each agent from stability?
            #shows the max and mean price change between consecutive snapshots
            #and how many of the 225 states still exceed the tolerance
            if len(ref_history_1) >= 2:
                prev1, curr1 = ref_history_1[-2], ref_history_1[-1]
                prev2, curr2 = ref_history_2[-2], ref_history_2[-1]
                diff1 = np.abs(curr1 - prev1)
                diff2 = np.abs(curr2 - prev2)
                print(f"         ref grid | ag1 max_dp={diff1.max():.4f} "
                      f"mean_dp={diff1.mean():.4f} states>{CONV_PRICE_TOL:.4f}: "
                      f"{(diff1 > CONV_PRICE_TOL).sum()}/225"
                      f" | ag2 max_dp={diff2.max():.4f} "
                      f"mean_dp={diff2.mean():.4f} states>{CONV_PRICE_TOL:.4f}: "
                      f"{(diff2 > CONV_PRICE_TOL).sum()}/225")

    #final metrics computed over the last EVAL_WINDOW steps
    prices  = np.array(prices)
    profits = np.array(profits)
    n_eval = min(EVAL_WINDOW, len(profits))
    avg_profit = profits[-n_eval:].mean(axis=0)   #per-firm average profit
    avg_price  = prices[-n_eval:].mean(axis=0)    #per-firm average price
    d = collusion_index(avg_profit[0], avg_profit[1])  #final collusion index

    #return all results and trained agents for post-training analysis
    return dict(delta=d, avg_price=avg_price, avg_profit=avg_profit,
                price_history=prices, profit_history=profits,
                converged_at=converged_at,
                agents=(ag1, ag2),
                alpha_log=np.array(alpha_log),
                delta_log=np.array(delta_log),
                greedy_price_log=np.array(greedy_price_log))


#PUNISHMENT TEST
#firm 1 (high-cost) deviates to its myopic best response, then both resume learned policy
def punishment_test(agents, price_history, n_steps=30):
    ag1, ag2 = agents

    #pre-deviation prices the last converged prices from training
    p1_star, p2_star = price_history[-1]

    #compute firm 1's static best response using its own cost C1
    res = minimize_scalar(lambda p: -profit(p, p2_star, C1),
                          bounds=(P_MIN, P_MAX), method="bounded")
    p1_dev = res.x

    p1, p2 = p1_star, p2_star
    traj = [(p1, p2)]  #record trajectory starting from pre-deviation state

    for step in range(n_steps):
        if step == 0:
            #firm 1 undercuts to maximise short-run profit
            new_p1 = p1_dev
        else:
            #then firm 1 plays its learned greedy policy
            s1 = np.array([normalise_price(p1), normalise_price(p2)])
            new_p1 = unnormalise_action(select_action(ag1, s1, deterministic=True))

        #firm 2 always plays its learned greedy policy (including in period 0)
        s2 = np.array([normalise_price(p2), normalise_price(p1)])
        new_p2 = unnormalise_action(select_action(ag2, s2, deterministic=True))

        p1, p2 = new_p1, new_p2
        traj.append((p1, p2))

    #returns (n_steps+1, 2) array pre-deviation state + n_steps of response
    return np.array(traj)


#plotting

def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    deltas = [r["delta"] for r in results]

    #histogram of collusion index across all sessions
    axes[0, 0].hist(deltas, bins=25, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].axvline(0, color="green", ls="--", label="Nash (Delta=0)")
    axes[0, 0].axvline(1, color="red",   ls="--", label="Monopoly (Delta=1)")
    axes[0, 0].set_xlabel("Collusion Index Delta"); axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Collusion Index (mean={np.mean(deltas):.3f})")
    axes[0, 0].legend()

    #price trajectory for session 0 (last 5000 steps), labelled with costs
    ph = results[0]["price_history"][-5000:]
    axes[0, 1].plot(ph[:, 0], alpha=0.7, lw=0.5, label="Firm 1 (c=1.0)")
    axes[0, 1].plot(ph[:, 1], alpha=0.7, lw=0.5, label="Firm 2 (c=0.75)")
    #separate nash and monopoly lines for each firm since they differ
    axes[0, 1].axhline(P_NASH1, color="green", ls="--", lw=0.8, label=f"Nash1 ({P_NASH1:.3f})")
    axes[0, 1].axhline(P_NASH2, color="green", ls=":",  lw=0.8, label=f"Nash2 ({P_NASH2:.3f})")
    axes[0, 1].axhline(P_MONO1, color="red",   ls="--", lw=0.8, label=f"Mono1 ({P_MONO1:.3f})")
    axes[0, 1].axhline(P_MONO2, color="red",   ls=":",  lw=0.8, label=f"Mono2 ({P_MONO2:.3f})")
    axes[0, 1].set_xlabel("Step (last 5000)"); axes[0, 1].set_ylabel("Price")
    axes[0, 1].set_title("Price Trajectory (Session 0)")
    axes[0, 1].legend(fontsize=7)

    #smoothed learning curve averaged across all sessions
    #average price across both firms at each step, then average across sessions
    max_len = max(r["price_history"].shape[0] for r in results)
    mean_prices = np.full(max_len, np.nan)
    counts = np.zeros(max_len)
    for r in results:
        ph_mean = r["price_history"].mean(axis=1)  #average of firm 1 and firm 2 per step
        n = len(ph_mean)
        mean_prices[:n] = np.nansum([mean_prices[:n], ph_mean], axis=0)
        counts[:n] += 1
    mean_prices = mean_prices / np.maximum(counts, 1)

    #smooth with a rolling window to reduce noise
    w = min(5000, max_len // 2)
    if w > 0:
        smoothed = np.convolve(mean_prices, np.ones(w) / w, mode="valid")
        axes[1, 0].plot(smoothed, color="steelblue", lw=1)
    axes[1, 0].axhline(P_NASH_AVG, color="green", ls="--", label="Nash (avg)")
    axes[1, 0].axhline(P_MONO_AVG, color="red",   ls="--", label="Monopoly (avg)")
    axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("Avg Price")
    axes[1, 0].set_title("Learning Curve"); axes[1, 0].legend()

    #punishment test on the first session with delta > 0.3
    collusive = [r for r in results if r["delta"] > 0.3]
    if collusive:
        traj = punishment_test(collusive[0]["agents"], collusive[0]["price_history"])
        steps = np.arange(-1, len(traj) - 1)
        axes[1, 1].plot(steps, traj[:, 0], "o-", ms=4, label="Firm 1 (deviator)")
        axes[1, 1].plot(steps, traj[:, 1], "s-", ms=4, label="Firm 2")
        axes[1, 1].axhline(P_NASH_AVG, color="green", ls="--")
        axes[1, 1].axhline(P_MONO_AVG, color="red",   ls="--")
        axes[1, 1].axvline(0, color="gray", ls=":", label="Deviation")
        axes[1, 1].set_xlabel("Period"); axes[1, 1].set_ylabel("Price")
        axes[1, 1].set_title("Punishment Response"); axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No collusive sessions (Delta > 0.3)", ha="center", va="center")

    plt.tight_layout()
    plt.savefig("sac_results.png", dpi=200)
    print("Saved sac_results.png")


#MAIN
if __name__ == "__main__":
    #printing initial (now shows per-firm benchmarks)
    print(f"p_nash1 = {P_NASH1:.4f}   p_nash2 = {P_NASH2:.4f}")
    print(f"p_mono1 = {P_MONO1:.4f}   p_mono2 = {P_MONO2:.4f}")
    print(f"pi_nash1 = {PI_NASH1:.4f}   pi_nash2 = {PI_NASH2:.4f}")
    print(f"pi_mono1 = {PI_MONO1:.4f}   pi_mono2 = {PI_MONO2:.4f}")
    print(f"p_range = [{P_MIN:.4f}, {P_MAX:.4f}]")
    print(f"conv_tol = {CONV_PRICE_TOL:.5f} (price units)")
    print(f"conv: policy stable at {N_REF}^2={N_REF**2} ref states "
          f"for {CONV_N_CHECKS}x{CONV_CHECK_EVERY//1000}k = "
          f"{CONV_N_CHECKS * CONV_CHECK_EVERY // 1000}k steps")
    print(f"alpha_freeze = step {ALPHA_FREEZE_STEP}")
    print(f"Running {N_SESSIONS} sessions x {T_MAX} steps max\n")

    total_start = time.time()
    results = []

    #printing session diagnostics
    for i in range(N_SESSIONS):
        print(f"-- Session {i} --")
        session_start = time.time()
        r = run_session(i)
        session_time = time.time() - session_start

        conv_msg = f"converged@{r['converged_at']}" if r["converged_at"] else "no convergence"
        print(f"  Delta={r['delta']:.3f}  prices=({r['avg_price'][0]:.4f}, "
              f"{r['avg_price'][1]:.4f})  {conv_msg}  [{session_time:.0f}s]\n")
        results.append(r)

    total_time = time.time() - total_start
    deltas = [r["delta"] for r in results]
    conv_count = sum(1 for r in results if r["converged_at"] is not None)

    print("=" * 55)
    print(f"Mean Delta       = {np.mean(deltas):.3f} +/- {np.std(deltas):.3f}")
    print(f"Median Delta     = {np.median(deltas):.3f}")
    print(f"Delta > 0.5      : {np.mean(np.array(deltas) > 0.5):.0%}")
    print(f"Delta > 0        : {np.mean(np.array(deltas) > 0):.0%}")
    print(f"Converged    : {conv_count}/{N_SESSIONS}")
    print(f"Total time   : {total_time / 60:.1f} minutes")

    #saving to CSV
    import csv
    print("Saving data")

    for i, r in enumerate(results):
        with open(f"sac_session_{i}_prices.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "price_1", "price_2", "profit_1", "profit_2"])
            for t, (pr, pi) in enumerate(zip(r["price_history"], r["profit_history"])):
                w.writerow([t, pr[0], pr[1], pi[0], pi[1]])

        with open(f"sac_session_{i}_logs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "delta", "alpha_1", "alpha_2", "greedy_p1", "greedy_p2"])
            for dl, al, gl in zip(r["delta_log"], r["alpha_log"], r["greedy_price_log"]):
                w.writerow([int(dl[0]), dl[1], al[1], al[2], gl[1], gl[2]])

        #saving .pt files for weights
        ag1, ag2 = r["agents"]
        torch.save({
            "agent1_actor": {k: v.cpu() for k, v in ag1["actor"].items()},
            "agent1_q1":    {k: v.cpu() for k, v in ag1["q1"].items()},
            "agent1_q2":    {k: v.cpu() for k, v in ag1["q2"].items()},
            "agent2_actor": {k: v.cpu() for k, v in ag2["actor"].items()},
            "agent2_q1":    {k: v.cpu() for k, v in ag2["q1"].items()},
            "agent2_q2":    {k: v.cpu() for k, v in ag2["q2"].items()},
            "agent1_log_alpha": ag1["log_alpha"].cpu(),
            "agent2_log_alpha": ag2["log_alpha"].cpu(),
        }, f"sac_agents_session_{i}.pt")

    with open("sac_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "delta", "avg_price_1", "avg_price_2",
                     "avg_profit_1", "avg_profit_2", "converged_at", "steps_run"])
        for i, r in enumerate(results):
            w.writerow([i, r["delta"], r["avg_price"][0], r["avg_price"][1],
                         r["avg_profit"][0], r["avg_profit"][1],
                         r["converged_at"] if r["converged_at"] else -1,
                         len(r["price_history"])])

    #saving hyperparameters
    with open("sac_hyperparams.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in [("c1", C1), ("c2", C2), ("a", A), ("a0", A0), ("mu", MU),
                      ("p_nash1", P_NASH1), ("p_nash2", P_NASH2),
                      ("p_mono1", P_MONO1), ("p_mono2", P_MONO2),
                      ("pi_nash1", PI_NASH1), ("pi_nash2", PI_NASH2),
                      ("pi_mono1", PI_MONO1), ("pi_mono2", PI_MONO2),
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
