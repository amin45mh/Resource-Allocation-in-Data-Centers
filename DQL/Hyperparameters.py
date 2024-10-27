class Hyperparameters():
    def __init__(self):
        self.RL_load_path = f'./final_weights.pth'
        self.save_path = f'./final_weights'
        self.learning_rate = 5e-4
        self.discount_factor = 0.9
        self.batch_size = 32
        self.targetDQN_update_rate = 10
        self.num_episodes = 1000         # TO EXPERIMENT
        self.num_test_episodes = 1000      # TO EXPERIMENT
        self.epsilon_decay = 0.999
        self.buffer_size = 10000
