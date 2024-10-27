import numpy as np
from tqdm import trange
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle


class Agent():
    def __init__(self, state_bins, n_actions, discount, lr, epsilon, env):
        self.gamma = discount
        self.alpha = lr
        self.epsilon = epsilon
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.env = env

        # Initialize the Q-table with zeros
        self.Q = np.zeros(state_bins + [n_actions])

        # Initialize lists to track rewards and losses
        self.collected_rewards = []
        self.loss_list = []

    def discretize_state(self, state):
        # Discretize the continuous state space
        discretized = []
        for i in range(len(state)):
            bins = np.linspace(0, 100, self.state_bins[i])
            discretized_index = int(np.digitize(state[i], bins) - 1)
            discretized_index = min(discretized_index, self.state_bins[i] - 1)  # Ensure the index is within bounds
            discretized.append(discretized_index)
        return tuple(discretized)

    def epsGreedy(self, Q, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(Q[s])

    def QLearning(self, n_episodes):
        K = trange(n_episodes)
        reward_array = np.zeros(n_episodes)
        returns = 0

        Q = self.Q

        for k in K:
            s = self.env.reset()
            s = self.discretize_state(s)

            terminated = False

            episode_reward = 0
            while not terminated:
                a = self.epsGreedy(Q, s)
                s_next, reward, terminated, _ = self.env.step(a)
                s_next = self.discretize_state(s_next)
                old_value = Q[s][a]
                next_max = np.max(Q[s_next])
                Q[s][a] += self.alpha * (reward + self.gamma * next_max - Q[s][a])

                # Track loss
                loss = abs(old_value - Q[s][a])
                self.loss_list.append(loss)

                episode_reward += reward

                if terminated:
                    K.set_description(f'Episode {k + 1} ended')
                    K.refresh()
                    reward_array[k] = episode_reward
                    break

                s = s_next

            self.collected_rewards.append(episode_reward)

        self.env.close()
        self.Q = Q

        return Q, reward_array

    def eval(self, n_episodes, bins, Q=None):
        if Q:
            self.Q = Q

        performance_metrics = {
            'avg_memory1_utilization': [],
            'avg_cpu1_utilization': [],
            'avg_memory2_utilization': [],
            'avg_cpu2_utilization': [],
            'blocking_probability': [],
            'episode_rewards': []
        }

        K = trange(n_episodes)

        for k in K:
            s = self.env.reset()
            s = self.discretize_state(s)

            terminated = False
            episode_reward = 0

            while not terminated:
                a = np.argmax(self.Q[s])
                s_next, reward, terminated, _ = self.env.step(a)
                s_next = self.discretize_state(s_next)
                episode_reward += reward
                s = s_next

            stats = self.env.get_statistics()
            performance_metrics['avg_memory1_utilization'].append(stats['avg_memory1_utilization'])
            performance_metrics['avg_cpu1_utilization'].append(stats['avg_cpu1_utilization'])
            performance_metrics['avg_memory2_utilization'].append(stats['avg_memory2_utilization'])
            performance_metrics['avg_cpu2_utilization'].append(stats['avg_cpu2_utilization'])
            performance_metrics['blocking_probability'].append(stats['blocking_probability'])
            performance_metrics['episode_rewards'].append(episode_reward)

            print(f"Episode: {k + 1}, Reward: {episode_reward:.2f}")

        avg_performance = {key: np.mean(value) for key, value in performance_metrics.items()}

        print("Evaluation Results:")
        for key, value in avg_performance.items():
            print(f"{key}: {value:.4f}")


        return performance_metrics

    def plot_learning_curves(self, bins):
        rewards_to_plot = self.collected_rewards

        # Calculate the Moving Average over the last 100 episodes or less if fewer episodes are available
        if len(rewards_to_plot) > 100:
            moving_average = np.convolve(rewards_to_plot, np.ones(100) / 100, mode='valid')
            moving_average_range = range(len(moving_average))
        else:
            moving_average = np.array(rewards_to_plot)
            moving_average_range = range(len(rewards_to_plot))

        rewards_to_plot = self.collected_rewards[:-100]
        # Plot Reward
        plt.figure()
        plt.title(f"Reward with {bins} Bins")
        plt.plot(rewards_to_plot, label='Reward', color='gray')
        if len(rewards_to_plot) > 100:
            plt.plot(moving_average_range, moving_average, label='Moving Average', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        # Save the figure
        plt.savefig(f'./Reward_vs_Episode_bins_{bins}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

        # Plot Loss
        plt.figure()
        plt.title(f"Loss with {bins} Bins")
        plt.plot(self.loss_list[:len(rewards_to_plot)], label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Training Loss")

        # Save the figure
        plt.savefig(f'./Loss_Rate_{bins}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()


# Functions to save and load Q-table
def saveQ(agent, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(agent.Q, outp, pickle.HIGHEST_PROTOCOL)


def loadAgent(state_bins, n_actions, discount, lr, epsilon, env, filename):
    agent = Agent(state_bins, n_actions, discount, lr, epsilon, env)
    with open(filename, 'rb') as inp:
        agent.Q = pickle.load(inp)
    return agent
