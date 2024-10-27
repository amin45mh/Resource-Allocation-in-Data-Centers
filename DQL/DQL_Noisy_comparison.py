import numpy as np
import matplotlib.pyplot as plt

# Load rewards
dqn_rewards = np.load('ddqn_rewards.npy')
noisy_rewards = np.load('noisyddqn_rewards.npy')

# Plot rewards
plt.figure()
plt.plot(dqn_rewards, label='DQN', color='blue')
plt.plot(noisy_rewards, label='Noisy DQN', color='red')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN vs Noisy DQN")
plt.legend()
plt.savefig(f'./DQN_vs_NoisyDQN.png', format='png', dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
plt.clf()
plt.close()