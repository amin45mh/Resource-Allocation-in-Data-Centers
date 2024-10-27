import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from collections import deque

from Hyperparameters import Hyperparameters

class Replay_Buffer():
    """
    Experience Replay Buffer to store experiences
    """
    def __init__(self, size, device):


        self.device = device
        
        self.size = size # size of the buffer
        
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.terminals = deque(maxlen=size)
        
        
    def store(self, state, action, next_state, reward, terminal):
        """
        Store experiences to their respective queues
        """      
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        
        
    def sample(self, batch_size):
        """
        Sample from the buffer
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor([self.terminals[i] for i in indices], dtype=torch.bool, device=self.device)

        return states, actions, next_states, rewards, terminals
    
    
    def __len__(self):
        return len(self.terminals)
    
class DQN(nn.Module):
    """
    The Deep Q-Network (DQN) model 
    """
    def __init__(self, num_actions, feature_size):
        super(DQN, self).__init__()
                                                         
        self.hidden1 = nn.Linear(feature_size, 16)      # TO DO EXPERIMENT
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(8, num_actions)

        
    def forward(self, x):
        
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x
        
class Agent:
    """
    Implementing Agent DQL Algorithm
    """
    
    def __init__(self, env:gym.Env, hyperparameters:Hyperparameters, device = False):
        
        # Some Initializations
        if not device:
            if torch.backends.cuda.is_built():
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif torch.backends.mps.is_built():
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters 
        self.epsilon = 0.99
        self.loss_list = []
        self.current_loss = 0
        self.episode_counts = 0

        self.action_space  = env.action_space
        self.feature_space = env.observation_space
        self.replay_buffer = Replay_Buffer(self.hp.buffer_size, device = self.device)
        
        # Initiate the online and Target DQNs
        self.onlineDQN = DQN(num_actions=self.action_space.n, feature_size=self.feature_space.shape[0]).to(self.device)
        self.targetDQN = DQN(num_actions=self.action_space.n, feature_size=self.feature_space.shape[0]).to(self.device)
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())
        self.targetDQN.eval()

        self.loss_function = nn.MSELoss()

        # set the optimizer to Adam and call it <self.optimizer>, i.e., self.optimizer = optim.Adam()
        self.optimizer = torch.optim.Adam(self.onlineDQN.parameters(), lr=self.hp.learning_rate)


    def epsilon_greedy(self, state):
        """
        Implement epsilon-greedy policy
        """
        # This function should return the action chosen by epsilon greedy algorithm # 
        if torch.rand(1).item() < self.epsilon:
            return self.action_space.sample()  # Random action
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.onlineDQN(state)
            return q_values.argmax().item()  # Greedy action
        
    def greedy(self, state):
        """
        Implement greedy policy
        """ 
        # This function should return the action chosen by greedy algorithm # 
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.onlineDQN(state)
        return q_values.argmax().item()  # Greedy action

    def apply_SGD(self, done):
        """
        Train DQN
            ended (bool): Indicates whether the episode meets a terminal state or not. If ended,
            calculate the loss of the episode.
        """ 
        
        # Sample from the replay buffer
        states, actions, next_states, rewards, terminals = self.replay_buffer.sample(self.hp.batch_size)    
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)       
      
        # Compute <Q_hat> using the online DQN
        Q_hat = self.onlineDQN(states).gather(1, actions)

        with torch.no_grad():   
            # Compute the maximum Q-value for off-policy update and call it <next_target_q_value> 
            next_q_values = self.targetDQN(next_states).max(1, keepdim=True)[0]
            next_target_q_value = rewards + (self.hp.discount_factor * next_q_values)
        
        next_target_q_value[terminals] = 0 # Set Q-value for terminal states to zero
        # Compute the Q-estimator and call it <y>
        y = next_target_q_value
        
        loss = self.loss_function(Q_hat, y) # Compute the loss
        
        # Update the running loss and learned counts for logging and plotting
        self.current_loss += loss.item()
        self.episode_counts += 1

        if done:
            episode_loss = self.current_loss / self.episode_counts # Average loss per episode
            # Track the loss for final graph
            self.loss_list.append(episode_loss) 
            self.current_loss = 0
            self.episode_counts = 0
        
        # Apply backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip the gradients
        # It's just to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), 2)
        
        # Update DQN by using the optimizer: <self.optimizer>
        self.optimizer.step()

    def apply_double_SGD(self, done):
        """
        Train Double DQN
            ended (bool): Indicates whether the episode meets a terminal state or not. If ended,
            calculate the loss of the episode.
        """ 
        
        # Sample from the replay buffer
        states, actions, next_states, rewards, terminals = self.replay_buffer.sample(self.hp.batch_size)    
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)       
        
        # Compute <Q_hat> using the online DQN
        Q_hat = self.onlineDQN(states).gather(1, actions)

        with torch.no_grad():   
            # Select next action using the online network
            next_actions = self.onlineDQN(next_states).argmax(1, keepdim=True)
            
            # Evaluate the action using the target network
            next_q_values = self.targetDQN(next_states).gather(1, next_actions)
            
            # Compute the target Q-value
            next_target_q_value = rewards + (self.hp.discount_factor * next_q_values)
        
        next_target_q_value[terminals] = 0 # Set Q-value for terminal states to zero
        # Compute the Q-estimator and call it <y>
        y = next_target_q_value
        
        loss = self.loss_function(Q_hat, y) # Compute the loss
        
        # Update the running loss and learned counts for logging and plotting
        self.current_loss += loss.item()
        self.episode_counts += 1

        if done:
            episode_loss = self.current_loss / self.episode_counts # Average loss per episode
            # Track the loss for final graph
            self.loss_list.append(episode_loss) 
            self.current_loss = 0
            self.episode_counts = 0
        
        # Apply backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip the gradients
        # It's just to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), 2)
        
        # Update DQN by using the optimizer: <self.optimizer>
        self.optimizer.step()


    def update_target(self):
        """
        Update the target network 
        """
        # Copy the online DQN into target DQN
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())

    
    def update_epsilon(self):
        """
        reduce epsilon by the decay factor
        """
        # Gradually reduce epsilon
        self.epsilon = max(0.01, self.epsilon * self.hp.epsilon_decay)
        

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention
        This can be used for later test of the trained agent
        """
        torch.save(self.onlineDQN.state_dict(), path)