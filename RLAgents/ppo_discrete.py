import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# CAP the standard deviation of the actor used in the paper
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_neurons=64, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.linear1 = nn.Linear(state_size + action_size, hidden_layer_neurons)
        self.linear2 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.linear3 = nn.Linear(hidden_layer_neurons, 1)
        
        self.linear4 = nn.Linear(state_size + action_size, hidden_layer_neurons)
        self.linear5 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.linear6 = nn.Linear(hidden_layer_neurons, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_neurons=64, seed=0, epsilon=1e-6, max_action=1.0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.action_size = action_size
        self.epsilon = epsilon
        self.max_action = max_action

        self.linear1 = nn.Linear(state_size, hidden_layer_neurons)
        self.linear2 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.mu = nn.Linear(hidden_layer_neurons, action_size)
        self.log_std = nn.Linear(hidden_layer_neurons, action_size)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_action(self, state, deterministic=False):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.max_action
        return action, log_prob, mu
    

class SACAgent:
    def __init__(self, observation_shape, action_size, max_action=1.0, seed=0, device='cpu', lr=0.0003, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, hidden_layer_neurons=256,
                 start_steps=10000, updates_per_step=1, target_update_interval=1, alpha=0.2, automatic_entropy_tuning=False):
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
        self.max_action = max_action
        self.seed = seed
        self.device = device
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.start_steps = start_steps
        self.updates_per_step = updates_per_step
        self.target_update_interval = target_update_interval
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Set seed for reproducability
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Actor and Critic Networks
        self.actor = Actor(self.state_size, self.action_size, hidden_layer_neurons=hidden_layer_neurons, seed=self.seed, max_action=self.max_action).to(self.device)
        self.critic = Critic(self.state_size, self.action_size, hidden_layer_neurons=hidden_layer_neurons, seed=self.seed).to(self.device)           # 2 critic networks
        self.critic_target = Critic(self.state_size, self.action_size, hidden_layer_neurons=hidden_layer_neurons, seed=self.seed).to(self.device)    # 2 target critic networks

        # Copy local critic parameters to target critic
        self.hard_update(self.critic, self.critic_target)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-4)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.seed)

        # Entropy coefficient / Entropy temperature
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.Tensor([self.action_size]).to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)

        self.actor.train()
        self.critic.train()
        self.critic_target.train()
    

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

    def act(self, state, step=None, deterministic=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action_values, _, mu = self.actor.get_action(state)
        self.actor.train()

        action = action_values.detach().cpu().numpy().squeeze()

        if step is None:
            if deterministic:
                action = mu.detach().cpu().numpy().squeeze()
            else:
                action = action_values.detach().cpu().numpy().squeeze()
        # Epsilon-greedy action selection
        elif step > self.start_steps:
            if deterministic:
                action = mu.detach().cpu().numpy().squeeze()
            else:
                action = action_values.detach().cpu().numpy().squeeze()
        else:
            action = 2*(np.random.rand(self.action_size)-0.5)*self.max_action
        if self.action_size==1:
            action = np.expand_dims(action, 0)
        return action
        
    def learn(self,):
        """Update value parameters using given batch of experience tuples."""

        # Obtain random minibatch of tuples from D
        if len(self.memory) > self.batch_size:

            actor_losses, critic_losses = [], []

            for gradient_step in range(self.updates_per_step):
                states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                
                states = torch.FloatTensor(states).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
                dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

                ## Compute and minimize the loss
                with torch.no_grad():
                    # Select action according to policy
                    next_state_action, next_state_log_pi, _ = self.actor.get_action(next_states)
                    # Compute the next Q values: min over all critics targets
                    qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action)
                    # add entropy term
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    # td error + entropy term
                    next_q_value = rewards + (1 - dones) * self.gamma * (min_qf_next_target)

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
                qf1, qf2 = self.critic(states, actions)

                # Compute critic loss
                qf1_loss = F.mse_loss(qf1, next_q_value)    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf2_loss = F.mse_loss(qf2, next_q_value)    # JQ = 𝔼(st,at)~D[0.5(Q2(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf_loss = qf1_loss+qf2_loss

                critic_losses.append(qf_loss.item())
                
                # Optimize the critic
                self.critic_optimizer.zero_grad()
                qf_loss.backward()
                self.critic_optimizer.step()
                
                # Action by the current actor for the sampled state
                pi, log_pi, _ = self.actor.get_action(states)

                # Compute actor loss
                # Min over all critic networks
                qf1_pi, qf2_pi = self.critic(states, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()         # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                actor_losses.append(actor_loss.item())
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

                    self.alpha = self.log_alpha.exp()

                # ------------------- soft update target network ------------------- #
                if gradient_step % self.target_update_interval == 0:
                    self.soft_update(self.critic, self.critic_target) 

            # print(np.mean(actor_losses), np.mean(critic_losses))



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


class PPOReplayBuffer:
    def __init__(self, batch_size, seed):
        random.seed(seed)
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []