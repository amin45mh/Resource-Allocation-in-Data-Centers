from Hyperparameters import Hyperparameters
from DQL import DQL
import numpy as np

if __name__ == '__main__':
    hyperparameters = Hyperparameters()    
    
    train = True
    #train = False
    
    #doubleDQN = True
    doubleDQN = False     # regular DQN
    
    #noisy_env = True
    noisy_env = False     # regular Env without noise
    
    # Run
    DRL = DQL(hyperparameters, train_mode=train, noise_mode=noisy_env) # Define the instance
    
    # Train
    if train:
        DRL.train(doubleDQN)
   
    else:
        DRL.evaluate()

        