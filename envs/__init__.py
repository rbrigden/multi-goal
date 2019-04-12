from gym import register
import gym


register(
    id='FetchPushSparse-v3',
    entry_point='gym.envs.robotics:FetchPushEnv',
    kwargs={
        'reward_type': 'sparse'
    },
    max_episode_steps=50,
)