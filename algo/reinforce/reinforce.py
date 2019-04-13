import argparse
import numpy as np
import gym
import ray
import ray.tune as tune

import torch
import torch.nn as nn
import torch.distributions
import torch.optim as optim
from torch.distributions import MultivariateNormal
from tensorboardX import SummaryWriter
import os


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight.data)

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
            nn.Linear(64, self.action_space)
        )

    def forward(self, input_x):
        loc = self.net(input_x)
        sigma = torch.eye(self.action_space) * self.std
        return MultivariateNormal(loc=loc, covariance_matrix=sigma.unsqueeze(0))

    def sample(self, inp):
        dist = self.forward(inp)
        sample = dist.sample()
        return sample

def proc_state(x):
    return x["observation"]


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, args, lr, gamma):
        self.args = args
        self.policy = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

    def train(self, env, render=False):
        self.optimizer.zero_grad()
        states, actions, rewards = self.generate_episode(env, render)
        Gt = self.compute_gt(states, rewards)
        loss = -torch.mean(Gt * self.policy(states).log_prob(actions))
        loss.backward()
        self.optimizer.step()

        return torch.sum(rewards).item(), loss.data, len(rewards)


    def generate_episode(self, env, render=False):
        states = []
        actions = []
        rewards = []
        state = proc_state(env.reset())

        terminal = False
        while not terminal:
            if render:
                env.render()
            with torch.no_grad():
                action = self.policy.sample(torch.FloatTensor(state.reshape(1, -1))).numpy().reshape((-1,))
            next_state, reward, terminal, info = env.step(action)
            next_state = proc_state(next_state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return torch.tensor(states, dtype=torch.float32), \
               torch.tensor(actions, dtype=torch.float32), \
               torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

    def compute_gt(self, states, rewards):
        episode_len = len(states)
        Gt = torch.zeros((episode_len, 1))
        discounts = torch.pow(self.gamma, torch.arange(0, episode_len, dtype=torch.float32))
        for t in range(episode_len):
            Gt[t] = torch.sum(rewards[t:] * discounts.view(-1, 1)[0:episode_len-t])
        return Gt

    def eval(self, n=100, render=False):
        returns = []
        for t in range(n):
            states, actions, rewards = self.generate_episode(env, render)
            returns.append(torch.sum(rewards).item())
        return np.mean(returns), np.max(returns), np.std(returns)

    def lr_step(self):
        self.lr_scheduler.step()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-2, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1, help="gamma")
    parser.add_argument('--checkpt', dest='checkpt', type=float,
                        default=1000, help="Checkpoint frequency")
    parser.add_argument('--resume', dest='resume', type=str,
                        default='', help="Checkpoint file name to resume from")
    parser.add_argument('--render', dest='render',
                         action='store_true', default=False,
                         help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()

def train(config):
    env = gym.make('FetchPushDense-v1')
    model = Policy(env.observation_space.spaces['observation'].shape[0], env.action_space.shape[0])
    model.apply(weight_init)
    writer = SummaryWriter(comment='_reinforce_')
    num_episodes = args.num_episodes
    lr = config["lr"]
    gamma = config["gamma"]
    render = args.render
    trainer = Reinforce(model, args, lr, gamma)

    start_epoch = 0
    if args.resume != '':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            trainer.policy.load_state_dict(checkpoint['state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    rewards_per_episode = []
    loss_per_episode = []
    steps_per_episode = []

    for epi in range(start_epoch, num_episodes):

        epi_rewards, epi_loss, len_epi = trainer.train(env)
        rewards_per_episode.append(epi_rewards)
        loss_per_episode.append(epi_loss)
        steps_per_episode.append(len_epi)

        if (epi + 1) % args.checkpt == 0:
            torch.save({'epoch': epi + 1,
            'state_dict': trainer.policy.state_dict(),
            'optimizer' : trainer.optimizer.state_dict(),
                        }, "reinf_changed_1e84_post50k.pt")

            mean_episode_reward = 1e2 * np.mean(rewards_per_episode)
            std_episode_reward = 1e2 * np.std(rewards_per_episode)
            mean_episode_loss = 1e2 * np.mean(loss_per_episode)
            avg_steps = np.mean(steps_per_episode)
            eval_mean_episode_return, eval_max_episode_return, eval_std_episode_return = [1e2 * x for x in
                                                                                          trainer.eval()]
            curr_lr = trainer.lr_scheduler.get_lr()[0]
            print('Episode: {}, Eval return: {}', epi, eval_mean_episode_return)
            writer.add_scalar('data/mean_episode_reward', mean_episode_reward, epi)
            writer.add_scalar('data/std_episode_reward', std_episode_reward, epi)
            writer.add_scalar('data/mean_episode_loss', mean_episode_loss, epi)
            writer.add_scalar('data/eval_mean_episode_return', eval_mean_episode_return, epi)
            writer.add_scalar('data/eval_std_episode_return', eval_std_episode_return, epi)
            writer.add_scalar('data/curr_lr', curr_lr, epi)
            writer.add_scalar('data/avg_steps', avg_steps, epi)

            rewards_per_episode = []
            loss_per_episode = []
            steps_per_episode = []

if __name__ == '__main__':
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes

    # Create the environment.
    env = gym.make('FetchPushDense-v1')

    env.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    config = {
                 "lr": 1e-3,
                 "gamma": 1
             }
    train(config)