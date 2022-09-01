"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from src.env import ArmEnv
# from src.rl import DDPG

MAX_EPISODES = 900
MAX_EP_STEPS = 200
ON_TRAIN = False

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
# rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()

actions = [(1,1), (0,1)]
def eval():
    # rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    i = 0
    while True:
        env.render()
        # a = rl.choose_action(s)
        try:
            a = actions[i]
            s, r, done = env.step(a)
            i+=1
        except IndexError:
            i=0
            continue
if ON_TRAIN:
    train()
else:
    eval()



