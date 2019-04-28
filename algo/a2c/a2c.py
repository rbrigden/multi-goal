import argparse
import numpy as np
import gym
import ray
import ray.tune as tune

import torch
import torch.nn as nn
import torch.distributions
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import os
import torch.nn.functional as F
import time

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight.data)


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.hidden_size = 64
        self.action_space = action_space
        self.observation_space = observation_space
        self.critic = Critic(self.hidden_size)
        self.policy = Policy(self.hidden_size, action_space)

        self.base_net = nn.Sequential(
            nn.Linear(observation_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.base_net(x)
        policy_out = self.policy(z)
        critic_out = self.critic(z)
        return policy_out, critic_out

    def value(self, x):
        return self.critic(self.base_net(x))

    def act(self, x):
        with torch.no_grad():
            action = self.policy(self.base_net(x)).sample()
        return action


class Critic(nn.Module):

    def __init__(self, input_space):
        super(Critic, self).__init__()
        self.input_space = input_space
        self.net = nn.Sequential(
            nn.Linear(self.input_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_x):
        return self.net(input_x)


class Policy(nn.Module):

    def __init__(self, obs_space, action_space, std=0.1):
        super(Policy, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.std = std
        self.net = nn.Sequential(
            nn.Linear(self.obs_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        )
        self.log_std = nn.Parameter(torch.zeros(self.action_space))

    def forward(self, input_x):
        loc = self.net(input_x)
        sigma = torch.exp(self.log_std)
        return Normal(loc=loc, scale=sigma)

    def sample(self, inp):
        dist = self.forward(inp)
        sample = dist.sample()
        return sample


class EnvManager:

    def __init__(self, env_name, num_envs, num_steps):
        self.envs = [gym.make(env_name) for _ in range(num_envs)]

        # obs_size = self.envs[0].observation_space.spaces['observation'].shape[0]
        # action_size = self.envs[0].action_space.shape[0]

        obs_size = self.envs[0].observation_space.shape[0]
        action_size = self.envs[0].action_space.shape[0]

        self.terminal_n = torch.zeros(num_steps + 1, num_envs, 1)
        self.state_n = torch.zeros(num_steps + 1, num_envs, obs_size)
        self.value_pred_n = torch.zeros(num_steps + 1, num_envs, 1)
        self.reward_n = torch.zeros(num_steps, num_envs, 1)
        self.action_n = torch.zeros(num_steps, num_envs, action_size)
        self.t = 0

        for idx, env in enumerate(self.envs):
            self.state_n[self.t, idx] = proc_state(env.reset())

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_size = obs_size
        self.action_size = action_size

    def reset(self):
        self.state_n[0].copy_(self.state_n[-1])
        self.terminal_n[0].copy_(self.terminal_n[-1])
        self.t = 0

    def step(self, actor_critic):

        for idx, env in enumerate(self.envs):
            if self.terminal_n[self.t, idx]:
                self.state_n[self.t, idx] = proc_state(env.reset())

        state_batch = self.state_n[self.t]

        with torch.no_grad():
            action_batch = actor_critic.act(state_batch)
            self.action_n[self.t] = action_batch

            value_batch = actor_critic.value(state_batch)
            self.value_pred_n[self.t] = value_batch

        for idx, env in enumerate(self.envs):
            next_state, reward, terminal, info = env.step(action_batch[idx].numpy())
            self.state_n[self.t + 1, idx] = proc_state(next_state)
            self.reward_n[self.t, idx] = reward
            self.terminal_n[self.t + 1, idx] = int(terminal)

        self.t += 1

    def compute_returns(self, actor_critic, gamma, gae_lambda, use_gae=True):
        returns = torch.zeros(self.num_steps, self.num_envs, 1)

        with torch.no_grad():
            next_value = actor_critic.value(self.state_n[-1])

            if use_gae:
                self.value_pred_n[-1] = next_value
                gae = 0
                for step in reversed(range(self.reward_n.size(0))):
                    delta = self.reward_n[step] + gamma * self.value_pred_n[step + 1] * self.terminal_n[step + 1] - \
                            self.value_pred_n[step]
                    gae = delta + gamma * gae_lambda * self.terminal_n[step + 1] * gae
                    returns[step] = gae + self.value_pred_n[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(self.reward_n.size(0))):
                    returns[step] = (returns[step + 1] * gamma * self.terminal_n[step + 1] + self.reward_n[step])

        return returns


def proc_state(x):
    # x = x["observation"]
    return torch.FloatTensor(x)


class A2C(object):

    def __init__(self, env_name, actor_critic, args):
        self.args = args
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.args.lr)
        self.eval_env = gym.make(env_name)
        self.manager = EnvManager(env_name, args.num_workers, args.num_steps)
        self.num_steps = args.num_steps

    def forward(self):

        for t in range(self.num_steps):
            self.manager.step(self.actor_critic)

        returns = self.manager.compute_returns(self.actor_critic, self.args.gamma, self.args.gae_lambda)

        action_out, value_out = self.actor_critic.forward(self.manager.state_n[:-1].view((-1, self.manager.obs_size)))

        value_loss = F.mse_loss(value_out, returns.view(-1, 1))
        action_loss = - torch.mean(
            action_out.log_prob(self.manager.action_n.view((-1, self.manager.action_size))) * returns.view((-1, 1)))

        loss = 0.5 * value_loss + 0.5 * action_loss
        return loss

    def train_step(self):

        self.optimizer.zero_grad()
        loss = self.forward()
        loss.backward()
        self.optimizer.step()
        self.manager.reset()


    def generate_episode(self, render=False):
        states = []
        actions = []
        rewards = []
        state = proc_state(self.eval_env.reset())

        terminal = False
        while not terminal:
            if render:
                self.eval_env.render()
                time.sleep(0.01)
            with torch.no_grad():
                action = self.actor_critic.act(state.view((1, -1))).numpy().reshape((-1,))
            next_state, reward, terminal, info = self.eval_env.step(action)
            next_state = proc_state(next_state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        self.eval_env.close()

        return torch.tensor(rewards).unsqueeze(1)

    def eval(self, n=100, render=False):
        returns = []
        for t in range(n):
            rewards = self.generate_episode(render)
            returns.append(torch.sum(rewards).item())
        return np.mean(returns), np.max(returns), np.std(returns)

    def lr_step(self):
        self.lr_scheduler.step()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-updates', type=int,
                        default=1000, help="Number of episodes to train on.")
    parser.add_argument('--num-steps', type=int,
                        default=100, help="Number of steps to take in each worker.")
    parser.add_argument('--num-workers', type=int,
                        default=20, help="Number of workers.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-4, help="The learning rate.")
    parser.add_argument('--gamma', type=float,
                        default=0.99, help="gamma")
    parser.add_argument('--gae-lambda', type=float,
                        default=0.95, help="gae_lambda")
    parser.add_argument('--checkpt', type=float,
                        default=5, help="Checkpoint frequency")
    parser.add_argument('--resume', type=str,
                        default='', help="Checkpoint file name to resume from")
    parser.add_argument('--render',
                        action='store_true', default=False,
                        help="Whether to render the environment.")

    return parser.parse_args()


def train(args):
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)
    # obs_size = env.observation_space.spaces['observation'].shape[0]
    # action_size = env.action_space.shape[0]

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    model = ActorCritic(obs_size, action_size)
    runner = A2C(env_name, model, args)

    for u in range(1, args.num_updates):
        # print("{}/{}".format(u, args.num_updates))
        runner.train_step()
        if u % args.checkpt == 0:
            print(model.policy.log_std)

            runner.eval(1, render=True)
            print(runner.eval(100 ))

    #
    # return
    # model.apply(weight_init)
    # writer = SummaryWriter(comment='_reinforce_')
    # num_episodes = args.num_episodes
    # lr = args.lr
    # gamma = args.gamma
    # render = args.render
    # trainer = A2C('FetchPushDense-v1', args.num_workers, model, args, lr, gamma)
    #
    # start_epoch = 0
    # if args.resume != '':
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         start_epoch = checkpoint['epoch']
    #         trainer.policy.load_state_dict(checkpoint['state_dict'])
    #         trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    #
    # rewards_per_episode = []
    # loss_per_episode = []
    # steps_per_episode = []
    #
    # for epi in range(start_epoch, num_episodes):
    #
    #     epi_rewards, epi_loss, len_epi = trainer.train(env)
    #     rewards_per_episode.append(epi_rewards)
    #     loss_per_episode.append(epi_loss)
    #     steps_per_episode.append(len_epi)
    #
    #     if (epi + 1) % args.checkpt == 0:
    #         torch.save({'epoch': epi + 1,
    #                     'state_dict': trainer.actor_critic.state_dict(),
    #                     'optimizer': trainer.optimizer.state_dict(),
    #                     }, "checkpoint.pt")
    #
    #         mean_episode_reward = np.mean(rewards_per_episode)
    #         std_episode_reward = np.std(rewards_per_episode)
    #         mean_episode_loss = np.mean(loss_per_episode)
    #         avg_steps = np.mean(steps_per_episode)
    #         eval_mean_episode_return, eval_max_episode_return, eval_std_episode_return = trainer.eval()
    #         print('Episode: {}, Eval return: {}'.format(epi, eval_mean_episode_return))
    #         writer.add_scalar('data/mean_episode_reward', mean_episode_reward, epi)
    #         writer.add_scalar('data/std_episode_reward', std_episode_reward, epi)
    #         writer.add_scalar('data/mean_episode_loss', mean_episode_loss, epi)
    #         writer.add_scalar('data/eval_mean_episode_return', eval_mean_episode_return, epi)
    #         writer.add_scalar('data/eval_std_episode_return', eval_std_episode_return, epi)
    #         writer.add_scalar('data/avg_steps', avg_steps, epi)
    #
    #         rewards_per_episode = []
    #         loss_per_episode = []
    #         steps_per_episode = []
    #

if __name__ == '__main__':
    # Parse command-line arguments.
    args = parse_arguments()
    train(args)
