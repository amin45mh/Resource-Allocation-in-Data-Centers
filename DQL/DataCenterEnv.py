import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DataCenterEnv(gym.Env):
    def __init__(self, max_steps_per_episode=100):          # Max steps set to 200 TO EXPERIMENT
        super(DataCenterEnv, self).__init__()

        # Actions: allocate resources or not
        self.action_space = spaces.Discrete(2)

        # State: CPU1, Memory1, CPU2, Memory2
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.int32)

        # Initialize resource capacities for the two data centers
        self.memory1 = 100
        self.cpu1 = 100
        self.memory2 = 100
        self.cpu2 = 100

        # Probability p, set to 0.1 
        self.pt = 0.1

        # Ongoing tasks
        self.tasks = []
        self.total_reward = 0
        self.state = self._get_state()
        self.total_tasks = 0
        self.blocked_tasks = 0
        
        # track steps of each episode
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0  # Initialize step count


    def _get_state(self):
        return np.array([self.memory1, self.cpu1, self.memory2, self.cpu2])

    def reset(self):
        # Reset the environment to the initial state
        self.memory1 = 100
        self.cpu1 = 100
        self.memory2 = 100
        self.cpu2 = 100
        self.tasks = []
        self.total_tasks = 0
        self.blocked_tasks = 0
        self.state = self._get_state()
        self.current_step = 0
        self.total_reward = 0
        return self.state

    def step(self, action):
        # # Update existing tasks (decrementing their remaining time?)
        new_tasks = []
        for mem, cpu, d in self.tasks:
            if d > 1:
                new_tasks.append((mem, cpu, d-1))
            else:
                if self.memory1 + mem <= 100 and self.cpu1 + cpu <= 100:
                    self.memory1 += mem
                    self.cpu1 += cpu
                elif self.memory2 + mem <= 100 and self.cpu2 + cpu <= 100:
                    self.memory2 += mem
                    self.cpu2 += cpu

        self.tasks = new_tasks

        # new task with 1-Pt probability
        if np.random.rand() > self.pt:
            # Generate random demand for memory and CPU
            demand_memory = np.random.randint(1, 35)
            demand_cpu = np.random.randint(1, 35)
            # Generate random duration for the task
            duration = np.random.randint(1, 11)

            # # Allocate resources to the task if available
            if action == 0:
                if self.memory1 >= demand_memory and self.cpu1 >= demand_cpu:
                    self.memory1 -= demand_memory
                    self.cpu1 -= demand_cpu
                    self.tasks.append((demand_memory, demand_cpu, duration))
                elif self.memory2 >= demand_memory and self.cpu2 >= demand_cpu:
                    self.memory2 -= demand_memory
                    self.cpu2 -= demand_cpu
                    self.tasks.append((demand_memory, demand_cpu, duration))
                else:
                    self.blocked_tasks += 1
            elif action == 1:
                if self.memory2 >= demand_memory and self.cpu2 >= demand_cpu:
                    self.memory2 -= demand_memory
                    self.cpu2 -= demand_cpu
                    self.tasks.append((demand_memory, demand_cpu, duration))
                elif self.memory1 >= demand_memory and self.cpu1 >= demand_cpu:
                    self.memory1 -= demand_memory
                    self.cpu1 -= demand_cpu
                    self.tasks.append((demand_memory, demand_cpu, duration))
                else:
                    self.blocked_tasks += 1

        # Update the state
        self.state = self._get_state()

        # reward:
        # Calculate free memory and CPU ratios
        rM1 = 1 - (self.memory1 / 100)
        rCPU1 = 1 - (self.cpu1 / 100)
        rM2 = 1 - (self.memory2 / 100)
        rCPU2 = 1 - (self.cpu2 / 100)

        # Calculate the minimum reward
        rM = min(rM1, rM2)
        rCPU = min(rCPU1, rCPU2)
        reward = min(rM, rCPU)

        self.total_reward += reward
        self.total_tasks += 1

        # Increment step count and check if episode should be done
        self.current_step += 1
        done = self.current_step >= self.max_steps_per_episode

        return self.state, reward, done, {}

    def render(self, mode='human', close=False):
        # Print the current state
        print(f"Data Center 1: Memory: {self.memory1} - CPU: {self.cpu1}")
        print(f"Data Center 2: Memory: {self.memory2} - CPU: {self.cpu2}")
        print(f"Total Reward: {self.total_reward}")
        print(f"Total Tasks: {self.total_tasks}")
        print(f"Blocked Tasks: {self.blocked_tasks}")
        print("Tasks:")
        for mem, cpu, d in self.tasks:
            print(f"Memory: {mem} - CPU: {cpu} - Duration: {d}")

        print("")
    
    def get_statistics(self):
        if self.total_tasks > 0:
            blocking_probability = self.blocked_tasks / self.total_tasks
        else:
            blocking_probability = 0
        avg_memory1_utilization = (100 - self.memory1) / 100
        avg_cpu1_utilization = (100 - self.cpu1) / 100
        avg_memory2_utilization = (100 - self.memory2) / 100
        avg_cpu2_utilization = (100 - self.cpu2) / 100

        return {
            'avg_memory1_utilization': avg_memory1_utilization,
            'avg_cpu1_utilization': avg_cpu1_utilization,
            'avg_memory2_utilization': avg_memory2_utilization,
            'avg_cpu2_utilization': avg_cpu2_utilization,
            'blocking_probability': blocking_probability
        }