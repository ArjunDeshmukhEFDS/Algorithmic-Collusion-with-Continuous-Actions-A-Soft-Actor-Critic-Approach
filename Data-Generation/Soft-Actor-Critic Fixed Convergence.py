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


DEVICE     = torch.device("cuda")
print(f"Using device: {DEVICE}")

#update parameter for how many steps per update
UPDATE_EVERY  = 4

#environment parameters
C     = 1.0       
#marginal cost
A     = 2.0       
#product quality
A0    = 0.0       
#outside option
MU    = 0.25      
#horizontal differentiation
DELTA = 0.95      
#discount factor

#Training configuration
#number of sessions
N_SESSIONS    = 20       
#maximum number of steps    
T_MAX         = 2_000_000
#evaluation window
EVAL_WINDOW   = 100_000    


#batch size
BATCH_SIZE    = 64
#minimum buffer size
MIN_BUFFER    = 1_000
#set seed for reproducibility
SEED_BASE     = 42

#SAC hyperparameters
#learning rate for all networks
LR            = 3e-4  
#soft update rate      
TAU           = 0.005     
#hidden layer size  
HIDDEN        = 64     
#target entropy is -dim(action)     
TARGET_ENTROPY = -1.0  
#initial entropy temperature     
INIT_ALPHA    = 0.2    
#gradient norm clipping    
GRAD_CLIP     = 1.0       
#freezing entropy tuning after this step  
ALPHA_FREEZE_STEP = 100_000 


#Convergence checking
#I check every 10k steps and require 10 consecutive stable checks = 100k steps.

#evaluate reference grid every N steps
CONV_CHECK_EVERY = 10_000  

#consecutive stable checks needed (10 × 10k = 100k steps)
CONV_N_CHECKS    = 5       

#max price change as fraction of price range
CONV_TOL         = 0.03     

#15×15 = 225 reference states (matches Calvano's 15² states)
N_REF            = 15       

#start decaying after alpha freeze
LR_DECAY_START = 500_000   

#LR reaches minimum by this step   
LR_DECAY_END   = 1_800_000 

#minimum learning rate is effectively zero   
LR_MIN         = 1e-6        


#Setting up the environment

#demand 
def demand(pi, pj):
    num = np.exp((A - pi) / MU)
    return num / (num + np.exp((A - pj) / MU) + np.exp(A0 / MU))

#profit function
def profit(pi, pj):
    return (pi - C) * demand(pi, pj)

#negative total industry profit (minimised to find joint-profit-maximising prices)
def neg_joint(prices):
    p1, p2 = prices
    return -profit(p1, p2) - profit(p2, p1)

#nash equilibrium via iterated BR 
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
    #Monopoly outcome via joint profit maximisation 
    res = minimize(neg_joint, [1.8, 1.8], bounds=[(C, C + 3)] * 2)
    p_mono = res.x[0] #symmetric, so both prices equal
    pi_mono = -neg_joint(res.x) / 2 #per-firm monopoly profit
    return p_nash, p_mono, pi_nash, pi_mono

#computing
P_NASH, P_MONO, PI_NASH, PI_MONO = compute_benchmarks()
SPREAD = P_MONO - P_NASH
P_MIN  = P_NASH - 0.3 * SPREAD
P_MAX  = P_MONO + 0.3 * SPREAD
CONV_PRICE_TOL = CONV_TOL * (P_MAX - P_MIN)   # absolute price tolerance


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



# Building 15x15 grid of reference states for convergence checking
# Each state is a (normalised own price, normalised opponent price) pair
_ref_prices = np.linspace(P_MIN, P_MAX, N_REF)
_ref_states_list = []
for p_own in _ref_prices:
    for p_opp in _ref_prices:
        _ref_states_list.append([normalise_price(p_own), normalise_price(p_opp)])
REF_STATES = torch.tensor(_ref_states_list, dtype=torch.float32, device=DEVICE)  # (225, 2)



#defining the xavier uniform for intialisation
def xavier_uniform(out_dim, in_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return (torch.rand(out_dim, in_dim, device=DEVICE) * 2 * limit - limit)

#small uniform initialisation for output layers (±0.003)
#prevents extreme initial Q-values or actions deep in tanh saturation
def small_uniform(out_dim, in_dim, bound=3e-3):
    return (torch.rand(out_dim, in_dim, device=DEVICE) * 2 * bound - bound)


#actor network: state -> (mean, log_std) for Gaussian policy
#architecture: 2 -> 64 -> 64 -> 2 (mu and log_sigma outputs)
def make_actor(state_dim=2, hidden=HIDDEN):
    params = {
        "l1_w": xavier_uniform(hidden, state_dim), #hidden layer 1 weights
        "l1_b": torch.zeros(hidden, device=DEVICE), 
        "l2_w": xavier_uniform(hidden, hidden),  #hidden layer 2 weights
        "l2_b": torch.zeros(hidden, device=DEVICE),
        "mu_w": small_uniform(1, hidden), #mean output head
        "mu_b": torch.zeros(1, device=DEVICE),
        "ls_w": small_uniform(1, hidden),  #log-std output head
        "ls_b": torch.zeros(1, device=DEVICE),
    }
    for v in params.values():
        v.requires_grad_(True)
    return params

#critic network: (state, action) -> Q-value
#architecture: 3 -> 64 -> 64 -> 1
def make_critic(state_dim=2, action_dim=1, hidden=HIDDEN):
    params = {
        "l1_w": xavier_uniform(hidden, state_dim + action_dim), #takes state + action
        "l1_b": torch.zeros(hidden, device=DEVICE),
        "l2_w": xavier_uniform(hidden, hidden),
        "l2_b": torch.zeros(hidden, device=DEVICE),
        "l3_w": small_uniform(1, hidden),          #scalar Q-value output
        "l3_b": torch.zeros(1, device=DEVICE),
    }
    for v in params.values():
        v.requires_grad_(True)
    return params


#forward pass through the actor network, takes normalised state and output mu and log std
#distribution over the raw (pre-squash) action space
def actor_forward(params, state):
    #two hidden layers with ReLU activation
    h = F.relu(state @ params["l1_w"].T + params["l1_b"])
    h = F.relu(h @ params["l2_w"].T + params["l2_b"])
    #mean head unbounded, represents the centre of the Gaussian
    mu = h @ params["mu_w"].T + params["mu_b"]
    #log-std head: clamped to [-20, 2] to prevent numerical instability
    #-20 gives near-deterministic policy (std ≈ 2e-9)
    #2 gives wide exploration (std ≈ 7.4)
    log_std = (h @ params["ls_w"].T + params["ls_b"]).clamp(-20, 2)
    return mu, log_std


#sample an action from the current policy and compute its log-probability
#target (entropy-regularised Bellman backup)
def actor_sample(params, state):
    mu, log_std = actor_forward(params, state)
    dist = Normal(mu, log_std.exp())
    #rsample uses the reparameterisation trick where u = mu + std * noise allows gradients to flow through the sample back to mu and std
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

# Forward pass through a critic network takes (state, action) and outputs a scalar Q-value estimate
def critic_forward(params, state, action):
    if action.dim() == 1:
        action = action.unsqueeze(-1)   #ensure action is (batch, 1) for concat
    x = torch.cat([state, action], dim=-1) #combine state and action as input
    h = F.relu(x @ params["l1_w"].T + params["l1_b"])
    h = F.relu(h @ params["l2_w"].T + params["l2_b"])
    return (h @ params["l3_w"].T + params["l3_b"]).squeeze(-1)

# Deep copy of network parameters for target networks
def copy_params(src):
    return {k: v.clone().detach().requires_grad_(False) for k, v in src.items()}

#Polyak averaging: target = (1 - τ) * target + τ * source  , slow update target networks
def soft_update(target, source, tau=TAU):
    for k in target:
        target[k].data.mul_(1 - tau).add_(tau * source[k].data)






#Convergence, get each agent's greedy (deterministic) price at all 225 reference states
def get_greedy_prices_at_ref(agent):
    with torch.no_grad():
        raw_actions = actor_deterministic(agent["actor"], REF_STATES)#(225,) actions in (-1,1)
        prices = P_MIN + (raw_actions + 1) * 0.5 * (P_MAX - P_MIN) #map to price interval
    return prices.cpu().numpy()

#Convergence check where policy has converged if, for both agents, the greedy price at every reference state has changed by less than CONV_PRICE_TOL
def check_convergence_calvano(ref_history_1, ref_history_2):
    if len(ref_history_1) < CONV_N_CHECKS:
        return False #not enough snapshots yet
    for hist in [ref_history_1, ref_history_2]:
        recent = np.array(hist[-CONV_N_CHECKS:])   #(N_CHECKS, 225)
        #For each of the 225 states, check max - min over the snapshots
        ranges = recent.max(axis=0) - recent.min(axis=0) 
        if ranges.max() > CONV_PRICE_TOL: #any state exceeds tolerance
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
    actor  = make_actor() #policy network: state -> (mu, log_std)
    q1     = make_critic() #twin critic 1: (state, action) -> Q-value
    q2     = make_critic() #twin critic 2: (state, action) -> Q-value
    q1_tgt = copy_params(q1) #target critic 1 (slowly updated copy for stable Bellman targets)
    q2_tgt = copy_params(q2) #target critic 2
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
                buffer=make_buffer()) #each agent gets its own replay buffer


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


#SAC update , updates critics, actor, entropy temperature and target networks
def update(agent, batch_size=BATCH_SIZE, gamma=DELTA, step=0):
    if len(agent["buffer"]) < batch_size:
        return #not enough experience to train yet
    
    #sample a random batch of transitions from replay buffer
    states, actions, rewards, next_states = buffer_sample(agent["buffer"], batch_size)
    alpha = agent["log_alpha"].exp().detach() #current entropy temperature (detached for critic/actor updates)

    #critic update
    #bellman target: r + γ * (min Q_target(s', a') - α * log pi(a'|s'))
    with torch.no_grad():
        next_a, next_lp = actor_sample(agent["actor"], next_states) #sample next action from current policy
        q1_next = critic_forward(agent["q1_tgt"], next_states, next_a) #target critic 1
        q2_next = critic_forward(agent["q2_tgt"], next_states, next_a)
        #min of two targets reduces overestimation
        target  = rewards + gamma * (torch.min(q1_next, q2_next) - alpha * next_lp)
        
    #update both critics by minimising MSE against the shared Bellman target    
    for qp, opt in [(agent["q1"], agent["q1_opt"]), (agent["q2"], agent["q2_opt"])]:
        loss = F.mse_loss(critic_forward(qp, states, actions), target)
        opt.zero_grad() #clears gradients from previous update
        loss.backward() #backprop
        clip_grads(qp.values()) #clipping gradiants
        opt.step()  #adam uses clipped gradients

    #actor update
    #maximise expected Q-value while maintaining entropy: min_k Q_k(s, a~) - α * log pi(a~|s)
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



#TRAINING LOOP
def run_session(session_id):
    #new seed for each session, 42 
    seed = SEED_BASE + session_id
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    #making agents
    ag1 = make_agent()
    ag2 = make_agent()
    
    #random initial prices within the allowed range
    p1 = np.random.uniform(P_MIN, P_MAX)
    p2 = np.random.uniform(P_MIN, P_MAX)

    prices, profits = [], []
    converged_at = None

    #Convergence tracking
    ref_history_1 = []   #store snapshots of each agent's greedy prices at the 225 reference states
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
            update(ag1, BATCH_SIZE, step=t)
            update(ag2, BATCH_SIZE, step=t)

            #Freeze entropy temperature at ALPHA_FREEZE_STEP and sync them
            if t + 1 == ALPHA_FREEZE_STEP:
                with torch.no_grad():
                    mean_log_alpha = (ag1["log_alpha"].item() + ag2["log_alpha"].item()) / 2
                    ag1["log_alpha"].fill_(mean_log_alpha)
                    ag2["log_alpha"].fill_(mean_log_alpha)
                print(f" Alpha frozen at {np.exp(mean_log_alpha):.4f} (mean of both agents)")
            
            #linear learning rate decay (analogous to epsilon decay in Q-learning)
            if t >= LR_DECAY_START:
               frac = min((t - LR_DECAY_START) / (LR_DECAY_END - LR_DECAY_START), 1.0)
               new_lr = LR * (1 - frac) + LR_MIN * frac #linear interpolation
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

            #compute rolling collusion index Δ from recent profits
            n_recent = min(EVAL_WINDOW, len(profits))
            recent = profits[-n_recent:]
            avg_pi = np.mean([x[0] + x[1] for x in recent]) / 2 #average per-firm profit
            d = collusion_index(avg_pi)

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
            frozen_msg = "" if t < ALPHA_FREEZE_STEP else " [α frozen]"
            print(f"  S{session_id} | t={t+1:>7d} | Δ={d:.3f} "
                  f"| α1={a1_alpha:.3f} α2={a2_alpha:.3f} "
                  f"| gp=({gp1_now:.4f},{gp2_now:.4f}) | {elapsed:.0f}s{frozen_msg}")

            #check convergence
            if converged_at is None:
                if check_convergence_calvano(ref_history_1, ref_history_2):
                    converged_at = t + 1
                    print(f"  Converged at step {converged_at} "
                          f"(policy stable at all {N_REF**2} ref states "
                          f"for {CONV_N_CHECKS * CONV_CHECK_EVERY // 1000}k steps)")
                    break
            #convergence diagnostics: how far is each agent from stability? ──
            #shows the max and mean price change between consecutive snapshots
            #and how many of the 225 states still exceed the tolerance
            if len(ref_history_1) >= 2:
                prev1, curr1 = ref_history_1[-2], ref_history_1[-1]
                prev2, curr2 = ref_history_2[-2], ref_history_2[-1]
                diff1 = np.abs(curr1 - prev1)
                diff2 = np.abs(curr2 - prev2)
                print(f"         ref grid | ag1 max_Δp={diff1.max():.4f} "
                      f"mean_Δp={diff1.mean():.4f} states>{CONV_PRICE_TOL:.4f}: "
                      f"{(diff1 > CONV_PRICE_TOL).sum()}/225"
                      f" | ag2 max_Δp={diff2.max():.4f} "
                      f"mean_Δp={diff2.mean():.4f} states>{CONV_PRICE_TOL:.4f}: "
                      f"{(diff2 > CONV_PRICE_TOL).sum()}/225")

     #final metrics computed over the last EVAL_WINDOW steps
    prices  = np.array(prices)
    profits = np.array(profits)
    n_eval = min(EVAL_WINDOW, len(profits))
    avg_profit = profits[-n_eval:].mean(axis=0)  #per-firm average profit
    avg_price  = prices[-n_eval:].mean(axis=0) #per-firm average price
    d = collusion_index(avg_profit.mean()) #final collusion index
    
    #return all results and trained agents for post-training analysis
    return dict(delta=d, avg_price=avg_price, avg_profit=avg_profit,
                price_history=prices, profit_history=profits,
                converged_at=converged_at,
                agents=(ag1, ag2),
                alpha_log=np.array(alpha_log),
                delta_log=np.array(delta_log),
                greedy_price_log=np.array(greedy_price_log))



#PUNISHMENT TEST 
def punishment_test(agents, price_history, n_steps=30):
    ag1, ag2 = agents
    
    #pre-deviation prices the last converged prices from training
    p1_star, p2_star = price_history[-1]
    
    #compute firm 1's static best response
    res = minimize_scalar(lambda p: -profit(p, p2_star),
                          bounds=(P_MIN, P_MAX), method="bounded")
    p1_dev = res.x

    p1, p2 = p1_star, p2_star
    traj = [(p1, p2)] #record trajectory starting from pre-deviation state

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
    
    #histogram of collusion index across all sessions
    deltas = [r["delta"] for r in results]
    axes[0, 0].hist(deltas, bins=25, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].axvline(0, color="green", ls="--", label="Nash (Δ=0)")
    axes[0, 0].axvline(1, color="red",   ls="--", label="Monopoly (Δ=1)")
    axes[0, 0].set_xlabel("Collusion Index Δ"); axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Collusion Index (mean={np.mean(deltas):.3f})")
    axes[0, 0].legend()
    
    #price trajectory for session 0 (last 5000 steps)
    ph = results[0]["price_history"][-5000:]
    axes[0, 1].plot(ph[:, 0], alpha=0.7, lw=0.5, label="Firm 1")
    axes[0, 1].plot(ph[:, 1], alpha=0.7, lw=0.5, label="Firm 2")
    axes[0, 1].axhline(P_NASH, color="green", ls="--", label=f"Nash ({P_NASH:.3f})")
    axes[0, 1].axhline(P_MONO, color="red",   ls="--", label=f"Mono ({P_MONO:.3f})")
    axes[0, 1].set_xlabel("Step (last 5000)"); axes[0, 1].set_ylabel("Price")
    axes[0, 1].set_title("Price Trajectory (Session 0)")
    axes[0, 1].legend(fontsize=8)
    
    #smoothed learning curve averaged across all sessions 
    #average price across both firms at each step, then average across sessions
    max_len = max(r["price_history"].shape[0] for r in results)
    mean_prices = np.full(max_len, np.nan)
    counts = np.zeros(max_len)
    for r in results:
        ph_mean = r["price_history"].mean(axis=1) #average of firm 1 and firm 2 per step
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
        axes[1, 1].plot(steps, traj[:, 1], "s-", ms=4, label="Firm 2")
        axes[1, 1].axhline(P_NASH, color="green", ls="--")
        axes[1, 1].axhline(P_MONO, color="red",   ls="--")
        axes[1, 1].axvline(0, color="gray", ls=":", label="Deviation")
        axes[1, 1].set_xlabel("Period"); axes[1, 1].set_ylabel("Price")
        axes[1, 1].set_title("Punishment Response"); axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No collusive sessions (Δ > 0.3)", ha="center", va="center")

    plt.tight_layout()
    plt.savefig("sac_results.png", dpi=200)
    print("Saved sac_results.png")



#MAIN
if __name__ == "__main__":
    #printing intial
    print(f"p_nash  = {P_NASH:.4f}   p_mono  = {P_MONO:.4f}")
    print(f"π_nash  = {PI_NASH:.4f}   π_mono  = {PI_MONO:.4f}")
    print(f"p_range = [{P_MIN:.4f}, {P_MAX:.4f}]")
    print(f"conv_tol = {CONV_PRICE_TOL:.5f} (price units)")
    print(f"conv: policy stable at {N_REF}²={N_REF**2} ref states "
          f"for {CONV_N_CHECKS}×{CONV_CHECK_EVERY//1000}k = "
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
        print(f"  Δ={r['delta']:.3f}  prices=({r['avg_price'][0]:.4f}, "
              f"{r['avg_price'][1]:.4f})  {conv_msg}  [{session_time:.0f}s]\n")
        results.append(r)

    total_time = time.time() - total_start
    deltas = [r["delta"] for r in results]
    conv_count = sum(1 for r in results if r["converged_at"] is not None)

    print("=" * 55)
    print(f"Mean Delta       = {np.mean(deltas):.3f} ± {np.std(deltas):.3f}")
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
        for k, v in [("c", C), ("a", A), ("a0", A0), ("mu", MU),
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
