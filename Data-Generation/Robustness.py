import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
import csv
from scipy.optimize import minimize_scalar, minimize


#sweep configuration
SESSIONS_PER_VARIANT = 3

#(label, hidden, lr, buffer_capacity) for each variant to test
VARIANTS = [
    ("Baseline",     64,  3e-4, 100_000),
    ("Hidden=32",    32,  3e-4, 100_000),
    ("Hidden=128",  128,  3e-4, 100_000),
    ("LR=1e-4",      64,  1e-4, 100_000),
    ("LR=1e-3",      64,  1e-3, 100_000),
    ("Buffer=50k",   64,  3e-4,  50_000),
    ("Buffer=200k",  64,  3e-4, 200_000),
]

#fixed parameters (same as baseline unless swept)
DEVICE         = torch.device("cuda")
C              = 1.0       #marginal cost
A              = 2.0       #product quality
A0             = 0.0       #outside option
MU             = 0.25      #horizontal differentiation
DELTA          = 0.95      #discount factor
T_MAX          = 2_000_000 #maximum number of steps
EVAL_WINDOW    = 100_000   #evaluation window
BATCH_SIZE     = 64
MIN_BUFFER     = 1_000
SEED_BASE      = 42
UPDATE_EVERY   = 4         #update parameter for how many steps per update
TAU            = 0.005     #soft update rate
TARGET_ENTROPY = -1.0      #target entropy is -dim(action)
INIT_ALPHA     = 0.2       #initial entropy temperature
GRAD_CLIP      = 1.0       #gradient norm clipping
ALPHA_FREEZE_STEP = 100_000   #freezing entropy tuning after this step
CONV_CHECK_EVERY  = 10_000    #evaluate reference grid every N steps
CONV_N_CHECKS     = 5         #consecutive stable checks needed
CONV_TOL          = 0.03      #max price change as fraction of price range
N_REF             = 15        #15x15 = 225 reference states
LR_DECAY_START    = 500_000   #start decaying after alpha freeze
LR_DECAY_END      = 1_800_000 #LR reaches minimum by this step
LR_MIN            = 1e-6      #minimum learning rate is effectively zero


#setting up the environment

#demand function
def demand(pi, pj):
    num = np.exp((A - pi) / MU)
    return num / (num + np.exp((A - pj) / MU) + np.exp(A0 / MU))

#profit function
def profit(pi, pj):
    return (pi - C) * demand(pi, pj)

#nash equilibrium via iterated BR and monopoly via joint profit maximisation
def compute_benchmarks():
    #initial guess
    p = 1.5
    for _ in range(1000):
        #find profit-maximising price given opponent charges p
        res = minimize_scalar(lambda pi, pj=p: -(pi - C) * demand(pi, pj),
                              bounds=(C, C + 3), method="bounded")
        #check if best response equals current price (fixed point)
        if abs(res.x - p) < 1e-10:
            break
        p = res.x
    p_nash = p
    pi_nash = profit(p_nash, p_nash)

    #negative total industry profit (minimised to find joint-profit-maximising prices)
    def neg_joint(prices):
        p1, p2 = prices
        return -profit(p1, p2) - profit(p2, p1)
    #monopoly outcome via joint profit maximisation
    res = minimize(neg_joint, [1.8, 1.8], bounds=[(C, C + 3)] * 2)
    p_mono = res.x[0]                  #symmetric, so both prices equal
    pi_mono = -neg_joint(res.x) / 2    #per-firm monopoly profit
    return p_nash, p_mono, pi_nash, pi_mono


#computing benchmarks
P_NASH, P_MONO, PI_NASH, PI_MONO = compute_benchmarks()
SPREAD = P_MONO - P_NASH
P_MIN  = P_NASH - 0.3 * SPREAD
P_MAX  = P_MONO + 0.3 * SPREAD
CONV_PRICE_TOL = CONV_TOL * (P_MAX - P_MIN)   #absolute price tolerance

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
#hidden size is parameterised since it varies across variants
def make_actor(hidden):
    params = {
        "l1_w": xavier_uniform(hidden, 2),       #hidden layer 1 weights
        "l1_b": torch.zeros(hidden, device=DEVICE),
        "l2_w": xavier_uniform(hidden, hidden),   #hidden layer 2 weights
        "l2_b": torch.zeros(hidden, device=DEVICE),
        "mu_w": small_uniform(1, hidden),          #mean output head
        "mu_b": torch.zeros(1, device=DEVICE),
        "ls_w": small_uniform(1, hidden),          #log-std output head
        "ls_b": torch.zeros(1, device=DEVICE),
    }
    for v in params.values():
        v.requires_grad_(True)
    return params

#critic network: (state, action) -> Q-value
#hidden size is parameterised since it varies across variants
def make_critic(hidden):
    params = {
        "l1_w": xavier_uniform(hidden, 3),         #takes state + action
        "l1_b": torch.zeros(hidden, device=DEVICE),
        "l2_w": xavier_uniform(hidden, hidden),
        "l2_b": torch.zeros(hidden, device=DEVICE),
        "l3_w": small_uniform(1, hidden),           #scalar Q-value output
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
def soft_update(target, source):
    for k in target:
        target[k].data.mul_(1 - TAU).add_(TAU * source[k].data)

#convergence, get each agent's greedy (deterministic) price at all 225 reference states
def get_greedy_prices_at_ref(agent):
    with torch.no_grad():
        raw = actor_deterministic(agent["actor"], REF_STATES)  #(225,) actions in (-1,1)
        prices = P_MIN + (raw + 1) * 0.5 * (P_MAX - P_MIN)    #map to price interval
    return prices.cpu().numpy()

#convergence check where policy has converged if, for both agents, the greedy price at every reference state has changed by less than CONV_PRICE_TOL
def check_convergence(ref_history_1, ref_history_2):
    if len(ref_history_1) < CONV_N_CHECKS:
        return False  #not enough snapshots yet
    for hist in [ref_history_1, ref_history_2]:
        recent = np.array(hist[-CONV_N_CHECKS:])
        if (recent.max(axis=0) - recent.min(axis=0)).max() > CONV_PRICE_TOL:  #any state exceeds tolerance
            return False
    return True


#replay buffer

#appending to buffer
def buffer_push(buf, state, action, reward, next_state):
    buf.append((state, action, reward, next_state))

#sampling from buffer
def buffer_sample(buf, batch_size):
    batch = random.sample(buf, batch_size)
    s, a, r, s2 = zip(*batch)
    return (torch.tensor(np.array(s),  dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(a),  dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(r),  dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(s2), dtype=torch.float32, device=DEVICE))


#creating the SAC agent
#parameterised by hidden size, learning rate and buffer capacity since these vary across variants
def make_agent(hidden, lr, buffer_cap):
    actor  = make_actor(hidden)       #policy network: state -> (mu, log_std)
    q1     = make_critic(hidden)      #twin critic 1: (state, action) -> Q-value
    q2     = make_critic(hidden)      #twin critic 2: (state, action) -> Q-value
    q1_tgt = copy_params(q1)          #target critic 1 (slowly updated copy for stable Bellman targets)
    q2_tgt = copy_params(q2)          #target critic 2
    #learnable entropy temperature, initialised to log(0.2)
    log_alpha = torch.tensor(np.log(INIT_ALPHA), device=DEVICE, requires_grad=True)
    #separate Adam optimiser for each component, using variant-specific lr
    actor_opt = torch.optim.Adam(actor.values(), lr=lr)
    q1_opt    = torch.optim.Adam(q1.values(), lr=lr)
    q2_opt    = torch.optim.Adam(q2.values(), lr=lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)
    return dict(actor=actor, q1=q1, q2=q2, q1_tgt=q1_tgt, q2_tgt=q2_tgt,
                log_alpha=log_alpha, actor_opt=actor_opt,
                q1_opt=q1_opt, q2_opt=q2_opt, alpha_opt=alpha_opt,
                buffer=deque(maxlen=buffer_cap))  #variant-specific buffer capacity

#choose an action given the current state (always stochastic in this sweep)
def select_action(agent, state):
    s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        a, _ = actor_sample(agent["actor"], s)
        return a.item()

#clip gradients, applied to actor and critic parameters after each backward
def clip_grads(params):
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        torch.nn.utils.clip_grad_norm_(grads, GRAD_CLIP)

#SAC update, updates critics, actor, entropy temperature and target networks
#base_lr is passed for LR decay calculation since it varies across variants
def update(agent, step, base_lr):
    if len(agent["buffer"]) < BATCH_SIZE:
        return  #not enough experience to train yet

    #sample a random batch of transitions from replay buffer
    states, actions, rewards, next_states = buffer_sample(agent["buffer"], BATCH_SIZE)
    alpha = agent["log_alpha"].exp().detach()  #current entropy temperature

    #critic update
    #bellman target: r + gamma * (min Q_target(s', a') - alpha * log pi(a'|s'))
    with torch.no_grad():
        next_a, next_lp = actor_sample(agent["actor"], next_states)  #sample next action from current policy
        q1_next = critic_forward(agent["q1_tgt"], next_states, next_a)  #target critic 1
        q2_next = critic_forward(agent["q2_tgt"], next_states, next_a)
        #min of two targets reduces overestimation
        target = rewards + DELTA * (torch.min(q1_next, q2_next) - alpha * next_lp)

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
#runs a single session with specific hidden size, learning rate and buffer capacity
def run_session(session_id, hidden, lr, buffer_cap):
    #new seed for each session
    seed = SEED_BASE + session_id
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    #making agents with variant-specific hyperparameters
    ag1 = make_agent(hidden, lr, buffer_cap)
    ag2 = make_agent(hidden, lr, buffer_cap)

    #random initial prices within the allowed range
    p1 = np.random.uniform(P_MIN, P_MAX)
    p2 = np.random.uniform(P_MIN, P_MAX)

    profits = []
    #store snapshots of each agent's greedy prices at the 225 reference states
    ref_history_1, ref_history_2 = [], []
    converged_at = None

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

        #compute profits from the logit demand model
        pi1, pi2 = profit(new_p1, new_p2), profit(new_p2, new_p1)
        #normalise rewards by Nash profit so r=1 means Nash, r>1 means supra-competitive
        r1, r2 = pi1 / PI_NASH, pi2 / PI_NASH

        #construct next states (prices just chosen become next period's observed state)
        next_state1 = np.array([normalise_price(new_p1), normalise_price(new_p2)])
        next_state2 = np.array([normalise_price(new_p2), normalise_price(new_p1)])

        #store (s, a, r, s') transitions in each agent's replay buffer
        buffer_push(ag1["buffer"], state1, a1, r1, next_state1)
        buffer_push(ag2["buffer"], state2, a2, r2, next_state2)

        #update networks every UPDATE_EVERY steps, once buffer has enough transitions
        if len(ag1["buffer"]) >= MIN_BUFFER and (t + 1) % UPDATE_EVERY == 0:
            #pass variant-specific base_lr for decay calculation
            update(ag1, t, lr)
            update(ag2, t, lr)

            #freeze entropy temperature at ALPHA_FREEZE_STEP and sync them
            if t + 1 == ALPHA_FREEZE_STEP:
                with torch.no_grad():
                    mean_la = (ag1["log_alpha"].item() + ag2["log_alpha"].item()) / 2
                    ag1["log_alpha"].fill_(mean_la)
                    ag2["log_alpha"].fill_(mean_la)

            #linear learning rate decay using the variant-specific base lr
            if t >= LR_DECAY_START:
                frac = min((t - LR_DECAY_START) / (LR_DECAY_END - LR_DECAY_START), 1.0)
                new_lr = lr * (1 - frac) + LR_MIN * frac  #linear interpolation from base lr
                for ag in [ag1, ag2]:
                    for opt in [ag["actor_opt"], ag["q1_opt"], ag["q2_opt"]]:
                        for pg in opt.param_groups:
                            pg["lr"] = new_lr

        #update current prices for the next period
        p1, p2 = new_p1, new_p2
        profits.append((pi1, pi2))

        #every CONV_CHECK_EVERY steps = snapshot + convergence check
        if (t + 1) % CONV_CHECK_EVERY == 0:
            #snapshot greedy prices at ALL 225 reference states for convergence tracking
            ref_history_1.append(get_greedy_prices_at_ref(ag1))
            ref_history_2.append(get_greedy_prices_at_ref(ag2))

            #check convergence
            if check_convergence(ref_history_1, ref_history_2):
                converged_at = t + 1
                break

            #print progress every 20k steps
            if (t + 1) % 20_000 == 0:
                n_recent = min(EVAL_WINDOW, len(profits))
                avg_pi = np.mean([x[0]+x[1] for x in profits[-n_recent:]]) / 2
                print(f"    t={t+1:>7d} Delta={collusion_index(avg_pi):.3f}")

    #final metrics computed over the last EVAL_WINDOW steps
    profits = np.array(profits)
    n_eval = min(EVAL_WINDOW, len(profits))
    avg_pi = profits[-n_eval:].mean(axis=0).mean()  #average per-firm profit
    return collusion_index(avg_pi), converged_at


#MAIN SWEEP
if __name__ == "__main__":
    print(f"Robustness sweep: {len(VARIANTS)} variants x {SESSIONS_PER_VARIANT} sessions "
          f"= {len(VARIANTS) * SESSIONS_PER_VARIANT} total runs\n")
    print(f"Benchmarks: p_nash={P_NASH:.4f}  p_mono={P_MONO:.4f}\n")

    all_results = []
    total_start = time.time()

    #loop over each variant and run SESSIONS_PER_VARIANT sessions
    for label, hidden, lr, buf_cap in VARIANTS:
        print(f"{'--'*25}")
        print(f"Variant: {label}  (hidden={hidden}, lr={lr}, buffer={buf_cap})")
        deltas = []
        converged = 0

        #printing session diagnostics
        for s in range(SESSIONS_PER_VARIANT):
            t0 = time.time()
            d, conv_at = run_session(s, hidden, lr, buf_cap)
            elapsed = time.time() - t0
            deltas.append(d)
            if conv_at is not None:
                converged += 1
            print(f"  Session {s}: Delta={d:.3f}  "
                  f"{'converged@'+str(conv_at) if conv_at else 'no convergence'}  "
                  f"[{elapsed:.0f}s]")

        mean_d = np.mean(deltas)
        std_d  = np.std(deltas)
        all_results.append((label, mean_d, std_d, converged, deltas))
        print(f"  -> Mean Delta = {mean_d:.3f} +/- {std_d:.3f}  "
              f"Converged: {converged}/{SESSIONS_PER_VARIANT}\n")

    total_time = time.time() - total_start

    #print summary table
    print("\n" + "=" * 60)
    print(f"{'Variant':<16} {'Mean D':>8} {'Std D':>8} {'Converged':>10}")
    print("-" * 60)
    for label, mean_d, std_d, conv, _ in all_results:
        print(f"{label:<16} {mean_d:>8.3f} {std_d:>8.3f} "
              f"{conv:>5}/{SESSIONS_PER_VARIANT}")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")

    #saving to CSV
    with open("robustness_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mean_delta", "std_delta", "converged",
                     "sessions"] + [f"delta_{i}" for i in range(SESSIONS_PER_VARIANT)])
        for label, mean_d, std_d, conv, deltas in all_results:
            w.writerow([label, f"{mean_d:.4f}", f"{std_d:.4f}",
                        f"{conv}/{SESSIONS_PER_VARIANT}",
                        SESSIONS_PER_VARIANT] + [f"{d:.4f}" for d in deltas])

    print("Saved robustness_table.csv")