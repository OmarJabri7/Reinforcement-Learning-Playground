#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 15:22:05 2021

@author: omar
"""

import gym


if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    
    env = gym.wrappers.Monitor(env, "recording")
    
    states = env.reset()
    
    total_rewards = 0.0
    
    total_steps = 0
    
    while True:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        total_steps+=1
        if done:
            break
print(f"Episode done in {total_steps} steps, total reward: {total_rewards}")