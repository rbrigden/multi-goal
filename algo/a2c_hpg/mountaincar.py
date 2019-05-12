from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import math
import numpy as np
import gym
from gym import register





class ContinuousMountainCarVarGoal(Continuous_MountainCarEnv):

    def __init__(self):
        super(ContinuousMountainCarVarGoal, self).__init__()
        self.goal_space = 1


    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        self.goal_reward = 100.0

        reward = 0
        if done:
            reward = 100.0
        reward -= math.pow(action[0], 2)*0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reward_query(self, action, state, goal):
        reward = 0
        position = state[0]
        velocity = state[1]

        reward -= math.pow(action[0], 2)*0.1
        if goal < 0:
            if position < goal and velocity < 0:
                reward += self.goal_reward
        else:
            if position > goal and velocity > 0:
                reward += self.goal_reward
        return reward




register(
    id='MountainCarContinuous-v5',
    entry_point='algo.a2c_hpg.mountaincar:ContinuousMountainCarVarGoal',
    max_episode_steps=999,
    reward_threshold=90.0,
)