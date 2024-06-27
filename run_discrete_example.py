# python run_discrete_example.py --exp=exp2 --agent=ddqn 
# python run_discrete_example.py --exp=exp2 --agent=dueling_dqn

import numpy as np
import pandas as pd
import os
import gymnasium as gym
import argparse
import torch
import random
from collections import deque

def get_args():
    parser = argparse.ArgumentParser('RL')
    parser.add_argument('--exp', type=str, default='exp1', 
                        help='Experiment folder name')
    parser.add_argument('--env', type=str, default='LunarLander-v2', 
                        help='RL environment name')
    parser.add_argument('--episodes', type=int, default=1000, 
                        help='number of episodes')
    parser.add_argument('--observation_dim', type=int, default=8, 
                        help='number of features in the observation space')
    parser.add_argument('--action_dim', type=int, default=4, 
                        help='number of possible actions')
    parser.add_argument('--render_mode', type=str, default=None, 
                        help='render mode for the environment')
    parser.add_argument('--agent', type=str, default='ddqn', 
                        help='RL agent')
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for reproducibility')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, 
                        help='epsilon decay rate (used if valid)')
    parser.add_argument('--epsilon_min', type=float, default=0.01, 
                        help='minimum epsilon value (used if valid)')
    parser.add_argument('--tau', type=float, default=0.01, 
                        help='tau for soft update (used if valid)')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='discount factor')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.00005, 
                        help='learning rate for training')
    parser.add_argument('--buffer_size', type=int, default=10000, 
                        help='size of the replay buffer')
    parser.add_argument('--norm', type=int, default=0, 
                        help='enable states normalization')
    args = parser.parse_args()
    return args

args = get_args()

if args.agent=='dueling_dqn':
    args.agent = 'DuelingDQNAgent'
    print('Using DuelingDQNAgent')
    from RLAgents.dueling_dqn import DuelingDQNAgent as Agent
else:
    args.agent = 'DDQNAgent'
    print('Using DDQNAgent')
    from RLAgents.ddqn import DDQNAgent as Agent

results_dir = f'results/{args.exp}'
checkpoints_dir = f'checkpoints/{args.exp}' #'checkpoints'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

name = f'env({args.env})_episode({args.episodes})_agent({args.agent})_tau({args.tau})'+\
        f'_gamma({args.gamma})_bs({args.batch_size})_lr({args.lr})_norm({int(args.norm)})'

summary_col_names = col_names = ['episode', 'step', 'step_in_episode',  'rewards', 'average_100_rewards']

# delete old results
filename = f'{results_dir}/{name}_summary.csv'
if(os.path.exists(filename) and os.path.isfile(filename)):
    os.remove(filename)
    print(f"Old {filename} file deleted")
else:
    print("Old results not found")

# delete old checkpoints
filenames = os.listdir(checkpoints_dir)
no_file = True
for ff in filenames:
    if name in ff:
        os.remove(f'{checkpoints_dir}/{ff}')
        print(f"Old {ff} file deleted")
        no_file = False
if no_file:
    print("Old checkpoints not found")


### Fix seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def norm(state, env_high, env_low):
    norm_states = 2*(state - env_low)/(env_high - env_low) - 1
    return norm_states


if __name__=='__main__':
    env = gym.make(args.env, render_mode=args.render_mode)
    env_high = env.observation_space.high
    env_low = env.observation_space.low

    agent = Agent(observation_shape=[args.observation_dim,], action_size=env.action_space.n, seed=0, device=args.device, lr=args.lr, hidden_layer_neurons=256)
    
    rewards_list = []                       # list containing scores from each episode
    rewards_window = deque(maxlen=100)      # last 100 scores
    eps = 1.0                               # initialize epsilon
    steps_n = 0                             # total number of steps
    best_rewards = -np.inf
    for episode in range(args.episodes):
        rewards = 0
        steps_in_episode = 0
        state, _ = env.reset()
        if args.norm:
            state = norm(state, env_high, env_low)
        while True:
            # action = np.random.randint(args.action_dim)
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            if args.norm:
                next_state = norm(next_state, env_high, env_low)
            done = bool(terminated or truncated)

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            rewards += reward
            steps_n += 1
            steps_in_episode += 1
            state = next_state

            if terminated or truncated:
                break 
        
        rewards_window.append(rewards)      # save most recent score
        rewards_list.append(rewards)        # save most recent score
        print(f'Episode {episode}/{args.episodes} ; eps: {eps} ; Rewards: {rewards} ; Average Rewards: {np.mean(rewards_window)}')

        eps = max(args.epsilon_min, args.epsilon_decay*eps)   # decrease epsilon
        
        result_df = pd.DataFrame([[episode, steps_n, steps_in_episode, rewards, np.mean(rewards_window)]], columns=summary_col_names, index=[pd.Timestamp(0).now()])
        hdr = False if os.path.isfile(f'{results_dir}/{name}_summary.csv') else True
        result_df.to_csv(f'{results_dir}/{name}_summary.csv', mode='a', index_label='Timestamp', header=hdr)

        if np.mean(rewards_window)>=best_rewards:
            best_rewards = np.mean(rewards_window)
            filenames = os.listdir(checkpoints_dir)
            for ff in filenames:
                if name in ff:
                    os.remove(f'{checkpoints_dir}/{ff}')
            agent.save_agent(f'{checkpoints_dir}/{name}_ep({episode})_rw({np.mean(rewards_window):.0f}).pth')