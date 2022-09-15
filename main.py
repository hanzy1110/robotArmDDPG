"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
import fire
import jax.numpy as jnp
import numpy as np
from src.env import ArmEnv
from src.inverseKinematics import inverseKControl
from src.newtonRaphsonMethod import NewtonControl, Tq
# from src.rl import DDPG

MAX_EPISODES = 900
MAX_EP_STEPS = 200
ON_TRAIN = False

# set env
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

actions = [(1,1,0), (0,1,0.5)]

def eval(x,y):
    goal = {'x':x, 'y':y, 'l':80}
    env = ArmEnv(goal)
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    # iKControl = inverseKControl(env.arm_info['l'])
    newtonControl = NewtonControl(env.arm_info['l'])

    # rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        # a = rl.choose_action(s)
        try:
            actual = env.arm_info["r"]
            a = newtonControl.get_action(env.goal, actual)
            print(f"Action : {a}")
            print(f"Actual : {actual}")
            print(f"Goal {env.goal}")
            s, r, done = env.step(a)
            if done:
                env.render()
                break
        except TypeError as e:
            print(e)
            raise(Exception)

# if ON_TRAIN:
#     train()
# else:
#     eval()
if __name__ == "__main__":

    theta_init = jnp.array([np.pi/6, np.pi/6, np.pi/6])
    ls = jnp.ones(3)*100
    goal=Tq(theta_init, ls)
    eval(goal[0], goal[1])
    # fire.Fire(eval)

