import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from Hyperparameters import Hyperparameters
from Agent import Agent
from DataCenterEnv import DataCenterEnv
from DataCenterNoisyEnv import DataCenterNoisyEnv

class DQL():
    def __init__(self, hyperparameters:Hyperparameters, train_mode, noise_mode):

        if train_mode:
            render = None
        else:
            render = "human"

        # Attention: <self.hp> contains all hyperparameters that you need
        self.hp = hyperparameters

        # Load the environment
        if noise_mode:
            self.env = DataCenterNoisyEnv()
        else:
            self.env = DataCenterEnv()
        
        # Initiate the Agent
        self.agent = Agent(env = self.env, hyperparameters = self.hp)
    
    
    def train(self, doubleDQN): 
        """                
        Traing the DQN via DQL
        """
        
        total_steps = 0
        self.collected_rewards = []
        
        # Training loop
        for episode in range(1, self.hp.num_episodes+1):
            
            state = self.env.reset()
            done = False
            step_size = 0
            episode_reward = 0
                                                
            while not done:           
                # find <action> via epsilon greedy 
                action = self.agent.epsilon_greedy(state)

                # find nest state and reward
                next_state, reward, done, info = self.env.step(action)
                
                #Print render
                self.env.render()
                
                # Put it into replay buffer
                self.agent.replay_buffer.store(state, action, next_state, reward, done)


                if len(self.agent.replay_buffer) > self.hp.batch_size:
                    
                    if(doubleDQN):
                        # use Double DQN
                        self.agent.apply_double_SGD(done)
                    else:
                        # use <self.agent.apply_SGD> implementation to update the online DQN
                        self.agent.apply_SGD(done)
                    
                    # Update target-network weights
                    if total_steps % self.hp.targetDQN_update_rate == 0:
                        # Copy the online DQN into the Target DQN using what you implemented in Class Agent
                        self.agent.update_target()

                state = next_state
                episode_reward += reward
                step_size +=1
                            
            self.collected_rewards.append(episode_reward)                     
            total_steps += step_size
                                                                           
            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()
                            
            # Print Results of the Episode
            printout = (f"Episode: {episode}, "
                      f"Total Time Steps: {total_steps}, "
                      f"Trajectory Length: {step_size}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.2f}")
            #print(printout)
        self.agent.save(self.hp.save_path + '.pth')
        self.plot_learning_curves()
        self.save_rewards()
        return self.collected_rewards
                                                                    

    def evaluate(self):  
        """                
        evaluate with the learned policy
        You can only run it if you already have trained the DQN and saved its weights as .pth file
        """
           
        # Load the trained DQN
        self.agent.onlineDQN.load_state_dict(torch.load(self.hp.RL_load_path, map_location=torch.device(self.agent.device)))
        self.agent.onlineDQN.eval()
        
        performance_metrics = {
            'avg_memory1_utilization': [],
            'avg_cpu1_utilization': [],
            'avg_memory2_utilization': [],
            'avg_cpu2_utilization': [],
            'blocking_probability': [],
            'episode_rewards': []
        }
        
        # Evaluating 
        for episode in range(1, self.hp.num_test_episodes+1):         
            state = self.env.reset()
            done = False
            step_size = 0
            episode_reward = 0
                                                           
            while not done:
                # Find the feature of <state> using your implementation <self.feature_representation>
                # state = self.feature_representation(state)
                state = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                # Act greedy and find <action> using what you implemented in Class Agent
                action = self.agent.greedy(state)

                next_state, reward, done, info = self.env.step(action)
                self.env.render()
                                
                state = next_state
                episode_reward += reward
                step_size += 1
            
            # Collect metrics after each episode
            stats = self.env.get_statistics()
            for key in performance_metrics:
                if key != 'episode_rewards':
                    performance_metrics[key].append(stats[key])
            performance_metrics['episode_rewards'].append(episode_reward)
            # Print Results of Episode            
            printout = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, ")
            print(printout)
        
        # Calculate average metrics
        avg_performance = {key: np.mean(value) for key, value in performance_metrics.items()}

        print("Evaluation Results:")
        for key, value in avg_performance.items():
            print(f"{key}: {value:.4f}")

        self.plot_evaluation_metrics(performance_metrics)
        pygame.quit()
        
    def plot_learning_curves(self):
        # Calculate the Moving Average over last 100 episodes
        rewards_to_plot = self.collected_rewards[:-100] if len(self.collected_rewards) > 100 else self.collected_rewards
        moving_average = np.convolve(self.collected_rewards, np.ones(100)/100, mode='valid')
        plt.figure()
        plt.title("Reward")
        plt.plot(rewards_to_plot, label='Reward', color='gray')
        plt.plot(moving_average, label='Moving Average', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Save the figure
        plt.savefig(f'./Reward_vs_Episode.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close() 
        
                
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_list, label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Training Loss")

       # Save the figure
        plt.savefig(f'./Learning_Curve.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def plot_evaluation_metrics(self, performance_metrics):
        """Plot the evaluation metrics collected during evaluation."""

        avg_reward = np.mean(performance_metrics['episode_rewards'])
        avg_memory1_utilization = np.mean(performance_metrics['avg_memory1_utilization'])
        avg_memory2_utilization = np.mean(performance_metrics['avg_memory2_utilization'])
        avg_cpu1_utilization = np.mean(performance_metrics['avg_cpu1_utilization'])
        avg_cpu2_utilization = np.mean(performance_metrics['avg_cpu2_utilization'])
        avg_blocking_probability = np.mean(performance_metrics['blocking_probability'])

        # Print average metrics
        print(f"Average Reward: {avg_reward:.4f}")
        print(f"Average Memory Utilization Data Center 1: {avg_memory1_utilization:.4f}")
        print(f"Average Memory Utilization Data Center 2: {avg_memory2_utilization:.4f}")
        print(f"Average CPU Utilization Data Center 1: {avg_cpu1_utilization:.4f}")
        print(f"Average CPU Utilization Data Center 2: {avg_cpu2_utilization:.4f}")
        print(f"Average Blocking Probability: {avg_blocking_probability:.4f}")

        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(performance_metrics['episode_rewards'], label='Episode Reward', color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(f'./Evaluation_Episode_Rewards.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Memory Utilization")
        plt.plot(performance_metrics['avg_memory1_utilization'], label='Data Center 1', color='green')
        plt.plot(performance_metrics['avg_memory2_utilization'], label='Data Center 2', color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Average Memory Utilization")
        plt.legend()
        plt.savefig(f'./Evaluation_Memory_Utilization.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("CPU Utilization")
        plt.plot(performance_metrics['avg_cpu1_utilization'], label='Data Center 1', color='green')
        plt.plot(performance_metrics['avg_cpu2_utilization'], label='Data Center 2', color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Average CPU Utilization")
        plt.legend()
        plt.savefig(f'./Evaluation_CPU_Utilization.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Blocking Probability")
        plt.plot(performance_metrics['blocking_probability'], label='Blocking Probability', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Blocking Probability")
        plt.legend()
        plt.savefig(f'./Evaluation_Blocking_Probability.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()
        
    def save_rewards(self):
        """ Save collected rewards as an npy file """
        np.save(self.hp.save_path + '_rewards.npy', self.collected_rewards)