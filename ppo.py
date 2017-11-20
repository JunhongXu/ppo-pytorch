def ppo_update():
    # value, action, logprob = net.forward(x)
    # ratio = exp(logprob - old_loprob)
    # advantage = vs - returns
    # surr1 = advantgae * ratio
    # surr2 = clamp(ratio, 1-epsilon, 1+epsilon) * advantage
    # policy_loss = min(surr1, surr2)
    # value_loss = mse(value, returns)
    # loss = value_loss + policy_loss
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    pass


def generate_trajectory(env, memory, policy, max_step=None):
    """generate a single trajectory using policy and insert the experiences into memory"""
    done = False
    nstep = 0
    obs = env.reset()
    while not done:
        # net
        pass