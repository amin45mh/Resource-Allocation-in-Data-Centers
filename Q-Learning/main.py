# Import necessary modules and functions
from Agent import Agent, saveQ, loadAgent
from DataCenterEnv import DataCenterEnv
import matplotlib.pyplot as plt
import numpy as np

# Initialize the environment
env = DataCenterEnv()

# Set the parameters
gamma = 0.9
alpha = 0.1
epsilon = 0.3

# Define the number of episodes for training
n_episodes = 1000
n_test = 300  # number of testing episodes

# List of bin configurations to test
bin_configurations = [2, 4, 8, 16]

# Dictionary to store Q-learning results
Q_ql_dict = {}
performance_metrics_dict = {}

# Loop through each bin configuration
for bins in bin_configurations:
    # Define state_bins for discretization of state space
    state_bins = [bins, bins, bins, bins]

    # Load Agent
    agent = Agent(state_bins, env.action_space.n, discount=gamma, lr=alpha, epsilon=epsilon, env=env)

    # Apply Q-Learning
    Q, reward_array = agent.QLearning(n_episodes)

    # Save the Q-function with a unique name
    filename = f'Qfunction_bins_{bins}.pkl'
    saveQ(agent, filename)

    # Save the learning curve plots without window size parameter
    agent.plot_learning_curves(bins)

    # Evaluate the trained agent
    agent = loadAgent(state_bins, env.action_space.n, discount=gamma, lr=alpha, epsilon=epsilon, env=env, filename=filename)
    performance_metrics = agent.eval(n_test, bins)
    performance_metrics_dict[bins] = performance_metrics

    # Store training rewards in the dictionary
    Q_ql_dict[bins] = {
        'training_rewards': reward_array
    }

# Plot the training rewards for different bin configurations
plt.figure(figsize=(10, 6))  # Increase the figure size for better visibility
for bins in bin_configurations:
    plt.plot([k + 1 for k in range(n_episodes)], Q_ql_dict[bins]['training_rewards'], label=f'{bins} bins')
plt.ylabel('Training Reward', fontsize=12)
plt.xlabel('Episode', fontsize=12)
plt.title(f'Learning by Q-Learning for {n_episodes} Episodes', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('./Q_learning_data_center_training_reward_vs_episode.png')
plt.show()

# Calculate and print average metrics for all bin configurations
overall_performance_metrics = {
    'avg_memory1_utilization': [],
    'avg_cpu1_utilization': [],
    'avg_memory2_utilization': [],
    'avg_cpu2_utilization': [],
    'blocking_probability': [],
    'episode_rewards': []
}

# Aggregate results from each bin configuration
for bins, metrics in performance_metrics_dict.items():
    for key in overall_performance_metrics:
        overall_performance_metrics[key].extend(metrics[key])

# Calculate average metrics across all bin configurations
avg_performance = {key: np.mean(value) for key, value in overall_performance_metrics.items()}

# Print overall evaluation results
print("Overall Evaluation Results:")
for key, value in avg_performance.items():
    print(f"{key}: {value:.4f}")

# Plot aggregated results
plt.figure()
plt.title("Aggregated Episode Rewards")
plt.plot(overall_performance_metrics['episode_rewards'], label='Episode Reward', color='blue')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig(f'./Aggregated_Evaluation_Episode_Rewards.png', format='png', dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
plt.clf()
plt.close()

plt.figure()
plt.title("Aggregated Memory Utilization")
plt.plot(overall_performance_metrics['avg_memory1_utilization'], label='Data Center 1', color='green')
plt.plot(overall_performance_metrics['avg_memory2_utilization'], label='Data Center 2', color='orange')
plt.xlabel("Episode")
plt.ylabel("Average Memory Utilization")
plt.legend()
plt.savefig(f'./Aggregated_Evaluation_Memory_Utilization.png', format='png', dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
plt.clf()
plt.close()

plt.figure()
plt.title("Aggregated CPU Utilization")
plt.plot(overall_performance_metrics['avg_cpu1_utilization'], label='Data Center 1', color='green')
plt.plot(overall_performance_metrics['avg_cpu2_utilization'], label='Data Center 2', color='orange')
plt.xlabel("Episode")
plt.ylabel("Average CPU Utilization")
plt.legend()
plt.savefig(f'./Aggregated_Evaluation_CPU_Utilization.png', format='png', dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
plt.clf()
plt.close()

plt.figure()
plt.title("Aggregated Blocking Probability")
plt.plot(overall_performance_metrics['blocking_probability'], label='Blocking Probability', color='red')
plt.xlabel("Episode")
plt.ylabel("Blocking Probability")
plt.legend()
plt.savefig(f'./Aggregated_Evaluation_Blocking_Probability.png', format='png', dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
plt.clf()
plt.close()
