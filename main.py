import envs
import gym
import numpy as np

if __name__ == '__main__':

    env = gym.make("FetchPushSparse-v3")
    x = env.reset()

    try:
        while True:
            env.render()
            env.step(np.random.randn(4,))

    except KeyboardInterrupt:
        print("Done")
    env.close()
