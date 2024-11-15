import random

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import gym
import d4rl

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=device)
    return x


def get_env_and_dataset(env_name):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    '''if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.'''

    for k, v in dataset.items():
        dataset[k] = torchify(v)
    return env, dataset


class ActionPred(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActionPred, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # self.log_std = log_std
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.actor(state)


class ActionTrain:
    def __init__(self, state_dim, action_dim, action_std_init, nets_num=1, lr_actor=1e-3):
        self.nets_num = nets_num
        self.action_nets = [ActionPred(state_dim, action_dim, action_std_init).to(device) for _ in range(nets_num)]
        self.MseLoss = nn.MSELoss(reduction='none')
        # self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [{'params': self.action_nets[i].parameters(), 'lr': lr_actor} for i in range(nets_num)], lr=lr_actor)

    def update(self, states, actions):
        self.optimizer.zero_grad()
        losses_list = [torch.mean(self.MseLoss(self.action_nets[i](states), actions), dim=-1) for i in
                       range(self.nets_num)]
        losses = torch.stack(losses_list).reshape((states.shape[0], -1)).to(device)
        loss_mean = torch.min(losses, dim=-1)

        loss = loss_mean.values.mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def update_init(self, states, actions, i):
        self.optimizer.zero_grad()
        loss = torch.mean(self.MseLoss(self.action_nets[i](states), actions), dim=-1)
        loss.mean().backward()
        self.optimizer.step()
        return loss

    '''def update(self, states, actions):
        self.optimizer.zero_grad()
        losses_list = [torch.mean(self.MseLoss(self.action_nets[i](states), actions), dim=-1) for i in range(self.nets_num)]
        losses = torch.stack(losses_list).reshape((states.shape[0], -1)).to(device)
        loss_mean = torch.min(losses, dim=-1)

        loss = loss_mean.values.mean()
        loss.backward()
        self.optimizer.step()
        return loss'''


def main(args):
    env, dataset = get_env_and_dataset(args.env_name)
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    action_train = ActionTrain(state_dim, action_dim, args.action_std_init)
    total_reward = 0

    if args.load_path != "":
        for i in range(action_train.nets_num):
            action_train.action_nets[i].load_state_dict(torch.load(args.load_path))
    step = 0
    while step < args.init_steps:
        for i in range(action_train.nets_num):
            sampled_indices = torch.randperm(dataset['actions'].size(0))[:args.batch_size]
            states = dataset['observations'][sampled_indices]
            actions = dataset['actions'][sampled_indices]
            total_reward += action_train.update_init(states, actions, i)
        step += 1
    step = 0
    while step < args.max_episode_steps:
        sampled_indices = torch.randperm(dataset['actions'].size(0))[:args.batch_size]
        states = dataset['observations'][sampled_indices]
        actions = dataset['actions'][sampled_indices]
        total_reward += action_train.update(states, actions)
        step += 1
        if step % args.save_interval == 0:
            print(f'Step: {step}, average reward: {total_reward / args.save_interval}')
            total_reward = 0
            for i in range(action_train.nets_num):
                torch.save(action_train.action_nets[i].state_dict(), args.save_path)
    return action_train.action_nets


def eval(args, nets):
    env = gym.make(args.env_name)
    step = 0
    state = env.reset()
    total_reward = 0
    while step < args.max_eval_steps:
        action = nets[random.randint(0, len(nets) - 1)](torch.Tensor(state).to(device))
        next_obs, reward, done, info = env.step(action.cpu().detach().numpy())
        total_reward += reward
        step += 1
        if done:
            print(f'Step: {step}, average reward: {total_reward}')
            total_reward = 0
            state = env.reset()
        else:
            state = next_obs


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser.add_argument('--env-name', default="kitchen-partial-v0")
    parser.add_argument('--env-name', default="kitchen-partial-v0")
    parser.add_argument('--max-episode-steps', default=5e4, type=int)
    parser.add_argument('--init-steps', default=100, type=int)
    parser.add_argument('--max-eval-steps', default=2e4, type=int)
    parser.add_argument('--action-std-init', default=0.6, type=float)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--save-interval', default=100, type=int)
    parser.add_argument('--save-path', default="../log/action_pred/model.pth", type=str)
    parser.add_argument('--load-path', default="", type=str)
    args = parser.parse_args()
    nets = main(args)
    eval(args, nets)
