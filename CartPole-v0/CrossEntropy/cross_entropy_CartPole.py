#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 19:22:12 2021

@author: omar
"""
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

envs = ["CartPole-v0", "FrozenLake-v0"]


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, states_size, hidden_size, actions_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Linear(states_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, actions_size))

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])


def iter_batch(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    state = env.reset()
    softmax = nn.Softmax(dim=1)
    while True:
        state_tensor = torch.FloatTensor([state])
        actions_proba = softmax(net(state_tensor))
        actions_proba = actions_proba.data.numpy()[0]
        action = np.random.choice(len(actions_proba), p=actions_proba)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(state=state, action=action)
        episode_steps.append(step)
        if done:
            episode = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(episode)
            episode_reward = 0.0
            episode_steps = []
            next_state = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        state = next_state


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_states = []
    train_actions = []

    for example in batch:
        if(example.reward < reward_bound):
            continue
        train_states.extend(map(lambda step: step.state, example.steps))
        train_actions.extend(map(lambda step: step.action, example.steps))

    train_states_tensor = torch.FloatTensor(train_states)
    train_actions_tensor = torch.LongTensor(train_actions)
    return train_states_tensor, train_actions_tensor, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make(envs[0])
    env = gym.wrappers.Monitor(
        env, directory=envs[0] + "-Recording", force=True)
    states_size = env.observation_space.shape[0]
    actions_size = env.action_space.n
    net = Net(states_size, HIDDEN_SIZE, actions_size)
    loss_fct = nn.CrossEntropyLoss()
    optim = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment=envs[0] + "-Writer")
    for iter_no, batch in enumerate(iter_batch(env, net, BATCH_SIZE)):
        states, actions, reward_bound, reward_mean = \
            filter_batch(batch, PERCENTILE)
        optim.zero_grad()
        action_scores = net(states)
        loss = loss_fct(action_scores, actions)
        loss.backward()
        optim.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" %
              (iter_no, loss.item(), reward_mean, reward_bound))
        writer.add_scalar("loss", loss.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        if reward_mean > 199:
            print("Solved!")
            break
    writer.close()
