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
import algo.a2c_hpg.mountaincar

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight.data)


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, goal_space, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.hidden_size = 64
        self.action_space = action_space
        # Add one for the goal
        self.observation_space = observation_space + goal_space
        self.critic = Critic(self.hidden_size)
        self.policy = Policy(self.hidden_size, action_space)

        self.base_net1 = nn.Sequential(
            nn.Linear(self.observation_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.base_net2 = nn.Sequential(
            nn.Linear(self.observation_space, hidden_size),
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

    def act(self, x):
        with torch.no_grad():
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


def append_goal_to_state(state_batch, goal):
    bsize = state_batch.size(0)
    goals = goal.repeat(bsize).view(bsize, 1)
    return torch.cat([state_batch, goals], dim=1)


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

    def __init__(self, env_name, num_envs, num_steps, as_storage=False):

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

        if not as_storage:
            for idx, env in enumerate(self.envs):
                self.state_n[self.t, idx] = proc_state(env.reset())

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_size = obs_size
        self.action_size = action_size
        self.as_storage = as_storage

    def reset(self):
        if self.as_storage:
            raise ValueError("This manager is being used as storage. Can't step.")

        self.state_n[0].copy_(self.state_n[-1])
        self.terminal_n[0].copy_(self.terminal_n[-1])
        self.t = 0

    def step(self, actor_critic, goal):
        if self.as_storage:
            raise ValueError("This manager is being used as storage. Can't step.")

        for idx, env in enumerate(self.envs):
            if self.terminal_n[self.t, idx]:
                self.state_n[self.t, idx] = proc_state(env.reset())

        state_batch = self.state_n[self.t]

        with torch.no_grad():
            action_batch = actor_critic.act(append_goal_to_state(state_batch, goal))
            self.action_n[self.t] = action_batch

            value_batch = actor_critic.value(append_goal_to_state(state_batch, goal))
            self.value_pred_n[self.t] = value_batch

        for idx, env in enumerate(self.envs):
            next_state, reward, terminal, info = env.step(action_batch[idx].numpy())
            self.state_n[self.t + 1, idx] = proc_state(next_state)
            self.reward_n[self.t, idx] = reward
            self.terminal_n[self.t + 1, idx] = int(terminal)

        self.t += 1

    def compute_returns(self, actor_critic, gamma, goals):
        returns = torch.zeros(self.num_steps+1, self.num_envs, 1)

        with torch.no_grad():
            # Different goal for each rollout
            expanded_goals = goals.unsqueeze(1)
            states_with_goals = torch.cat([self.state_n[-1], expanded_goals], dim=1)
            next_value = actor_critic.value(states_with_goals)
            returns[-1] = next_value
            for step in reversed(range(self.reward_n.size(0))):
                returns[step] = (returns[step + 1] * gamma * (1 - self.terminal_n[step + 1]) + self.reward_n[step])

        return returns

def compute_returns(states_with_goals, rewards, terminals, actor_critic, gamma):
    returns = torch.zeros(len(states_with_goals), 1)

    with torch.no_grad():
        states_with_goals = states_with_goals
        next_value = actor_critic.value(states_with_goals[-1])
        returns[-1] = next_value
        for step in reversed(range(len(rewards))):
            returns[step] = (returns[step + 1] * gamma * (1 - terminals[step + 1]) + rewards[step])

    return returns

def proc_state(x):
    # x = x["observation"]
    return torch.FloatTensor(x)


class A2C(object):

    def __init__(self, env_name, actor_critic, args):
        self.args = args
        self.env_name = env_name

        assert args.num_active_goals % args.num_workers == 0

        self.actor_critic = actor_critic
        self.optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=self.args.lr, alpha=0.99, eps=1e-5)
        self.eval_env = gym.make(env_name)

        # TODO: This is currently hardcoded for mountaincar will need to update
        self.original_goal = torch.tensor([0.45])

        # original goal manager
        self.og_manager = EnvManager(env_name, args.num_workers, args.num_steps)

        # active goal managers
        self.ag_manager = EnvManager(self.env_name, self.args.num_active_goals, self.args.num_steps)

        self.num_steps = args.num_steps

    def rollout(self):

        # Generate rollouts under the original goal
        for t in range(self.num_steps):
            self.og_manager.step(self.actor_critic, self.original_goal)

        # Collect all feasible active goals
        feasible_active_goals = self.og_manager.state_n.clone()
        num_active_goals_per_worker = self.args.num_active_goals // self.args.num_workers


        # Valid new trajectories
        original_trajectory_map = []
        valid_trajectory_idxs = []

        active_goal_idxs = []
        active_goals = []
        for i in range(self.args.num_workers):

            for j in range(num_active_goals_per_worker):

                for k in np.random.permutation(self.num_steps):

                    original_rollout_idx = i
                    new_rollout_idx = num_active_goals_per_worker * i + j

                    # Sample a new active goal
                    worker_goal_idx = np.random.randint(0, self.num_steps)

                    # Goal is the position (state index 0)
                    worker_active_goal = feasible_active_goals[worker_goal_idx, original_rollout_idx, 0]

                    if worker_active_goal < 0.1:
                        continue

                    # rewards under the new goal

                    original_env = self.og_manager.envs[original_rollout_idx]
                    new_rewards_and_terminals = [original_env.goal_query(self.og_manager.action_n[t, original_rollout_idx],
                                                                         self.og_manager.state_n[t, original_rollout_idx],
                                                                         worker_active_goal) for t in range(self.args.num_steps)]

                    new_rewards, new_terminals = zip(*new_rewards_and_terminals)

                    # check if we have a long enough trajectory
                    if new_terminals.index(True) <= self.args.min_trajectory_length:
                        continue

                    print("OG rollout {}, AG rollout {}, Attempts{}, Length {}".format(i, j, k, new_terminals.index(True)))

                    new_rewards = torch.tensor(new_rewards).float().view(-1, 1)
                    new_terminals = torch.tensor([0.0] + list(new_terminals)).float().view(-1, 1)


                    original_states = self.og_manager.state_n[:, original_rollout_idx].clone()
                    original_actions = self.og_manager.action_n[:, original_rollout_idx].clone()

                    # update the state value predictions under the new goal
                    with torch.no_grad():
                        new_value_preds = self.actor_critic.value(append_goal_to_state(original_states, worker_active_goal))


                    self.ag_manager.state_n[:, new_rollout_idx].copy_(original_states)
                    self.ag_manager.action_n[:, new_rollout_idx].copy_(original_actions)
                    self.ag_manager.reward_n[:, new_rollout_idx].copy_(new_rewards)
                    self.ag_manager.terminal_n[:, new_rollout_idx].copy_(new_terminals)
                    self.ag_manager.value_pred_n[:, new_rollout_idx].copy_(new_value_preds)
                    active_goal_idxs.append(worker_goal_idx)
                    active_goals.append(worker_active_goal)
                    original_trajectory_map.append(original_rollout_idx)
                    valid_trajectory_idxs.append(new_rollout_idx)
                    break

        # Do some cleanup
        # self.ag_manager.state_n = torch.index_select(self.ag_manager.state_n, 1, torch.tensor(valid_trajectory_idxs))
        # self.ag_manager.action_n = torch.index_select(self.ag_manager.action_n, 1, torch.tensor(valid_trajectory_idxs))
        # self.ag_manager.reward_n = torch.index_select(self.ag_manager.reward_n, 1, torch.tensor(valid_trajectory_idxs))
        # self.ag_manager.terminal_n = torch.index_select(self.ag_manager.terminal_n, 1, torch.tensor(valid_trajectory_idxs))
        # self.ag_manager.value_pred_n = torch.index_select(self.ag_manager.value_pred_n, 1, torch.tensor(valid_trajectory_idxs))
        # self.ag_manager.num_envs = len(valid_trajectory_idxs)


        return torch.tensor(active_goal_idxs).long(), \
               torch.tensor(active_goals).float(), \
               original_trajectory_map, \
               valid_trajectory_idxs

    def compute_active_goal_loss(self, active_goals, original_trajectory_map, valid_trajectory_idxs, og_action_log_probs):

        # A tensor of goals that correspond to the rollouts considered under these goals
        # expanded_ags = active_goals.unsqueeze(1).unsqueeze(0).repeat(self.args.num_steps+1, 1, 1)
        og_action_log_probs = og_action_log_probs.view(self.args.num_steps, self.args.num_workers, 1)

        ag_value_losses, ag_policy_losses = [], []
        for j in range(active_goals.size(0)):
            i = valid_trajectory_idxs[j]

            active_goal = active_goals[j]

            # Append the appropriate goal to every state
            states_with_active_goals = torch.cat([self.ag_manager.state_n[:, i], active_goal.repeat(self.num_steps+1).view(-1, 1)], dim=1)

            ag_flat_action_dists, ag_flat_value_preds = self.actor_critic.forward(states_with_active_goals[:-1].view((-1, self.ag_manager.obs_size + 1)))

            ag_action_log_probs = ag_flat_action_dists.log_prob(self.ag_manager.action_n[:, i].view((-1, self.ag_manager.action_size)))

            traj_og_action_log_probs = og_action_log_probs[:, original_trajectory_map[j]]


            traj_rewards = self.ag_manager.reward_n[:, i]
            traj_terminals = self.ag_manager.terminal_n[:, i]
            ag_returns = compute_returns(states_with_active_goals, traj_rewards, traj_terminals, self.actor_critic, self.args.gamma)


            mask = traj_terminals.clone()
            mask = (1 - (torch.cumsum(mask, dim=0) > 0)).float()

            ag_advantages = (ag_returns[:-1] - ag_flat_value_preds) * mask[:-1]
            ag_value_loss = ag_advantages.pow(2).mean()

            log_action_ratios = ag_action_log_probs - traj_og_action_log_probs
            action_weights = torch.exp(torch.cumsum(log_action_ratios, dim=0))

            ag_policy_loss = - torch.mean(mask[:-1] * (ag_advantages.detach() * action_weights * ag_action_log_probs))

            ag_policy_losses.append(ag_policy_loss)
            ag_value_losses.append(ag_value_loss)

        if len(ag_policy_losses) == 0:
            return 0, 0

        return sum(ag_policy_losses) / len(ag_policy_losses), sum(ag_value_losses) / len(ag_value_losses)


    def forward(self):
        active_goal_idxs, active_goals, original_trajectory_map, valid_trajectory_idxs = self.rollout()

        og_goals = torch.ones(self.args.num_workers) * self.original_goal
        og_returns = self.og_manager.compute_returns(self.actor_critic, self.args.gamma, og_goals)

        expanded_ogs = og_goals.unsqueeze(1).unsqueeze(0).repeat(self.args.num_steps + 1, 1, 1)
        states_with_og_goal = torch.cat([self.og_manager.state_n, expanded_ogs], dim=2)

        og_flat_action_dists, og_flat_value_preds = self.actor_critic.forward(states_with_og_goal[:-1].view((-1, self.og_manager.obs_size+1)))


        og_action_log_probs = og_flat_action_dists.log_prob(self.og_manager.action_n.view((-1, self.og_manager.action_size))).view(self.args.num_steps, self.args.num_workers, 1)
        ag_policy_loss, ag_value_loss = self.compute_active_goal_loss(active_goals, original_trajectory_map, valid_trajectory_idxs, og_action_log_probs.detach())

        og_advantages = og_returns[:-1] - og_flat_value_preds.view(self.args.num_steps, self.args.num_workers, 1)
        og_value_loss = og_advantages.pow(2).mean()

        policy_entropy = og_flat_action_dists.entropy().mean()

        og_policy_loss = - torch.mean(og_advantages.detach() * og_action_log_probs)

        alpha = self.args.num_workers / (self.args.num_workers + len(active_goal_idxs))

        policy_loss = alpha * og_policy_loss + (1 - alpha) * ag_policy_loss
        value_loss = alpha * og_value_loss + (1 - alpha) * ag_value_loss

        return value_loss, policy_loss, -policy_entropy

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
        self.og_manager.reset()

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
                time.sleep(0.01)
            with torch.no_grad():
                action = self.actor_critic.act(append_goal_to_state(state.view((1, -1)), self.original_goal)).numpy().reshape((-1,))
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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-updates', type=int,
                        default=1000, help="Number of episodes to train on.")
    parser.add_argument('--min-trajectory-length', type=int,
                        default=20, help="Number of episodes to train on.")
    parser.add_argument('--num-active-goals', type=int,
                        default=5, help="Number of episodes to train on.")
    parser.add_argument('--num-steps', type=int,
                        default=100, help="Number of steps to take in each worker.")
    parser.add_argument('--num-workers', type=int,
                        default=20, help="Number of workers.")
    parser.add_argument('--lr',  type=float,
                        default=1e-4, help="The learning rate.")
    parser.add_argument('--entropy-coef',  type=float,
                        default=0.05, help="Entropy coefficient.")
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

    model = ActorCritic(obs_size, action_size, goal_space=1)
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
