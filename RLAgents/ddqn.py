
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layer_neurons=64, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.linear1 = nn.Linear(state_size, hidden_layer_neurons)
        self.linear2 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.linear3 = nn.Linear(hidden_layer_neurons, action_size) 
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDQNAgent():
    def __init__(self, observation_shape, action_size, seed=0, device='cpu', lr=0.0005, buffer_size=10000, batch_size=125, gamma=0.99, 
                 tau=0.01, hidden_layer_neurons=64, max_grad_norm=None):
        """Initialize an Agent object.
        
        Params
        ======
            observation_shape ([int,]): shape of the observation space; 1D: [int,]; 2D: [int,int] (could be expanded to CNN)
            action_size (int): dimension of each action
            seed (int): random seed
            device (str): device used cpu/cuda
            lr (float): learning rate
            buffer_size (int): replay buffer size
            batch_size (int): batch size for learning
            gamma (float): discount factor
            tau (float): interpolation parameter for soft update
            hidden_layer_neurons (int): Q-network hidden layer neurons
        """
        self.observation_shape = observation_shape
        self.state_size = observation_shape[0]
        self.action_size = action_size
        self.seed = seed
        self.device = device
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        # Set seed for reproducability
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, hidden_layer_neurons, seed=self.seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, hidden_layer_neurons, seed=self.seed).to(self.device)
        self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.seed)
    

    def remember(self, state, action, reward, next_state, done):
        """Add sample to the replay buffer.
        
        Params
        ======
            state (Numpy ndarray of states): states
            actions (int): actions
            reward (float): step reward
            next_state (Numpy ndarray of states): next_state
            done (bool): done due to terminal state or truncated
        """
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, ):
        """Update value parameters using given batch of experience tuples."""

        # Obtain random minibatch of tuples from D
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
            
            ### Calculate expected Q-value from local network
            q_expected = self.qnetwork_local(states).gather(1, actions)

            with torch.no_grad():
                ### Extract next maximum estimated value from target network
                q_targets_next = q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)

            # 1-step TD target from bellman equation
            q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
            
            ### Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(q_expected, q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # ------------------- soft update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
  
    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_agent(self, filepath):
        torch.save({
            'Qnetwork': self.qnetwork_local.state_dict()
        }, filepath)

    def load_agent(self, filepath):
        checkpoint = torch.load(filepath)
        self.qnetwork_local.load_state_dict(checkpoint['Qnetwork'])
        self.qnetwork_target.load_state_dict(checkpoint['Qnetwork'])
        self.qnetwork_local.train()
        self.qnetwork_target.train()

class ReplayBuffer:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)