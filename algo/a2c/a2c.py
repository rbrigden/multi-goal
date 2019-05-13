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
import algo.a2c.seek

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

        self.base_net1 = nn.Sequential(
            nn.Linear(observation_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.base_net2 = nn.Sequential(
            nn.Linear(observation_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        z1 = self.base_net1(x)
        z2 = self.base_net2(x)

        policy_out = self.policy(z1)
        critic_out = self.critic(z2)
        return policy_out, critic_out

    def value(self, x):
        return self.critic(self.base_net2(x))

    def act(self, x, deterministic=False):
        with torch.no_grad():
            if deterministic:
                action = self.policy(self.base_net1(x)).mean
            else:
                action = self.policy(self.base_net1(x)).sample()
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
            nn.Linear(64, 1)
        )

    def forward(self, input_x):
        return self.net(input_x)




def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


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
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.action_mean = init_(nn.Linear(64, self.action_space))

        self.log_std = nn.Parameter(torch.zeros(self.action_space))

    def forward(self, input_x):
        z = self.net(input_x)
        loc = self.action_mean(z)
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

    def compute_returns(self, actor_critic, gamma):
        returns = torch.zeros(self.num_steps+1, self.num_envs, 1)

        with torch.no_grad():
            next_value = actor_critic.value(self.state_n[-1])
            returns[-1] = next_value
            for step in reversed(range(self.reward_n.size(0))):
                returns[step] = (returns[step + 1] * gamma * (1 - self.terminal_n[step + 1]) + self.reward_n[step])

        return returns


def proc_state(x):
    # x = x["observation"]
    return torch.FloatTensor(x)


class A2C(object):

    def __init__(self, env_name, actor_critic, args):
        self.args = args
        self.actor_critic = actor_critic
        self.optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=self.args.lr, alpha=0.99, eps=1e-5)
        self.eval_env = gym.make(env_name)
        self.manager = EnvManager(env_name, args.num_workers, args.num_steps)
        self.num_steps = args.num_steps
        self.lr_scheduler = optim.lr_scheduler.LambdaLR

    def forward(self):

        for t in range(self.num_steps):
            self.manager.step(self.actor_critic)

        returns = self.manager.compute_returns(self.actor_critic, self.args.gamma)
        action_out, value_out = self.actor_critic.forward(self.manager.state_n[:-1].view((-1, self.manager.obs_size)))

        advantages = returns[:-1] - value_out.view(self.args.num_steps, self.args.num_workers, 1)

        value_loss = advantages.pow(2).mean()

        policy_entropy = action_out.entropy().mean()

        action_log_probs = action_out.log_prob(self.manager.action_n.view((-1, self.manager.action_size))).view(self.args.num_steps, self.args.num_workers, self.manager.action_size)
        action_loss = - torch.mean(advantages.detach() * action_log_probs)

        return value_loss, action_loss, -policy_entropy

    def train_step(self, warmup=False):

        self.optimizer.zero_grad()
        value_loss, action_loss, entropy_loss = self.forward()

        if warmup:
            value_loss.backward()
        else:
            (self.args.value_coef * value_loss + action_loss + self.args.entropy_coef * entropy_loss).backward()

        if self.args.max_grad_norm:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.manager.reset()

        return value_loss.item(), action_loss.item(), entropy_loss.item()

    def generate_episode(self, render=False):
        states = []
        actions = []
        rewards = []
        state = proc_state(self.eval_env.reset())

        terminal = False
        while not terminal:
            if render:
                self.eval_env.render()
                time.sleep(0.2)
            with torch.no_grad():
                action = self.actor_critic.act(state.view((1, -1)), deterministic=render).numpy().reshape((-1,))
            next_state, reward, terminal, info = self.eval_env.step(action)
            next_state = proc_state(next_state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # self.eval_env.close()

        return torch.tensor(rewards).unsqueeze(1)

    def eval(self, n=100, render=False):
        returns = []
        for t in range(n):
            rewards = self.generate_episode(render)
            returns.append(torch.sum(rewards).item())
        return np.mean(returns), np.max(returns), np.std(returns)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-updates', type=int,
                        default=1000, help="Number of episodes to train on.")
    parser.add_argument('--num-steps', type=int,
                        default=100, help="Number of steps to take in each worker.")
    parser.add_argument('--num-workers', type=int,
                        default=20, help="Number of workers.")
    parser.add_argument('--lr',  type=float,
                        default=1e-4, help="The learning rate.")
    parser.add_argument('--entropy-coef',  type=float,
                        default=0.01, help="Entropy coefficient.")
    parser.add_argument('--value-coef', type=float,
                        default=0.5, help="Value coefficient")
    parser.add_argument('--max-grad-norm', type=float,
                        default=0.5, help="Max gradient norm")
    parser.add_argument('--gamma', type=float,
                        default=0.99, help="gamma")
    parser.add_argument('--gae-lambda', type=float,
                        default=0.95, help="gae_lambda")
    parser.add_argument('--checkpt', type=float,
                        default=5, help="Checkpoint frequency")
    parser.add_argument('--resume', type=str,
                        default=None, help="Checkpoint file name to resume from")
    parser.add_argument('--env', type=str,
                        default="Pendulum-v0", help="environment name")
    parser.add_argument('--render',
                        action='store_true', default=False,
                        help="Whether to render the environment.")

    return parser.parse_args()


def train(args):


    env_name = args.env
    env = gym.make(env_name)
    # obs_size = env.observation_space.spaces['observation'].shape[0]
    # action_size = env.action_space.shape[0]

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    model = ActorCritic(obs_size, action_size)
    runner = A2C(env_name, model, args)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            runner.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.render:
        runner.eval(5, render=True)
        exit(0)
    else:
        writer = SummaryWriter(comment='_a2c_')

    for u in range(1, args.num_updates):

        total_num_steps = u * args.num_workers * args.num_steps
        value_loss, action_loss, entropy_loss = runner.train_step(warmup=False)

        writer.add_scalar('value_loss', value_loss, total_num_steps)
        writer.add_scalar('action_loss', action_loss, total_num_steps)
        writer.add_scalar('entropy_loss', entropy_loss, total_num_steps)

        if u % args.checkpt == 0:

            mean_eval_return, max_eval_return, std_eval_return = runner.eval(20)
            writer.add_scalar('eval_mean_return', mean_eval_return, total_num_steps)
            writer.add_scalar('eval_max_return', max_eval_return, total_num_steps)
            writer.add_scalar('eval_std_return', std_eval_return, total_num_steps)

            torch.save({'state_dict': runner.actor_critic.state_dict(),
                        'optimizer': runner.optimizer.state_dict(),
                        }, "a2c_checkpoint.pt")


if __name__ == '__main__':
    # Parse command-line arguments.
    args = parse_arguments()
    train(args)
