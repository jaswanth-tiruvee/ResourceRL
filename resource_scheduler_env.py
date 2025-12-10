"""
Custom Gymnasium Environment for Resource Scheduling
Simulates a server environment where an RL agent learns to optimize job priority allocation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict


class ResourceSchedulerEnv(gym.Env):
    """
    Custom environment for resource scheduling optimization.
    
    State Space:
        - server_load: Current server utilization (0.0 to 1.0)
        - queue_length: Number of jobs waiting in queue (normalized)
        - avg_job_size: Average size of jobs in queue (normalized)
        - current_priority: Current priority level being processed
    
    Action Space:
        - priority_level: Discrete action [0, 1, 2, 3, 4] representing priority levels
          0 = Lowest, 4 = Highest
    
    Reward:
        - Based on latency: lower latency = higher reward
        - Penalizes high server load and long queues
        - Rewards efficient priority allocation
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None, num_servers: int = 3):
        super().__init__()
        
        self.num_servers = num_servers
        self.max_queue_length = 50
        self.max_job_size = 100
        
        # Action space: 5 priority levels (0=lowest, 4=highest)
        self.action_space = spaces.Discrete(5)
        
        # State space: [server_load, queue_length, avg_job_size, current_priority]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Internal state
        self.server_load = np.random.uniform(0.3, 0.7)
        self.queue_length = np.random.randint(5, 20)
        self.job_queue = []
        self.current_priority = 2  # Default medium priority
        self.episode_steps = 0
        self.max_episode_steps = 200
        
        # Job processing parameters
        self.job_arrival_rate = 0.3
        self.job_processing_rate = 0.4
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation (state)."""
        avg_job_size = np.mean([job['size'] for job in self.job_queue]) if self.job_queue else 0.0
        
        return np.array([
            self.server_load,
            min(self.queue_length / self.max_queue_length, 1.0),
            min(avg_job_size / self.max_job_size, 1.0),
            self.current_priority / 4.0  # Normalize to [0, 1]
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional info for debugging."""
        return {
            "queue_length": self.queue_length,
            "server_load": self.server_load,
            "current_priority": self.current_priority,
            "episode_steps": self.episode_steps
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.server_load = np.random.uniform(0.3, 0.7)
        self.queue_length = np.random.randint(5, 20)
        self.job_queue = [
            {
                'size': np.random.uniform(10, self.max_job_size),
                'priority': np.random.randint(0, 5),
                'arrival_time': 0
            }
            for _ in range(self.queue_length)
        ]
        self.current_priority = 2
        self.episode_steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.episode_steps += 1
        
        # Update current priority based on action
        self.current_priority = action
        
        # Simulate job arrivals
        if np.random.random() < self.job_arrival_rate:
            new_job = {
                'size': np.random.uniform(10, self.max_job_size),
                'priority': np.random.randint(0, 5),
                'arrival_time': self.episode_steps
            }
            self.job_queue.append(new_job)
            self.queue_length = len(self.job_queue)
        
        # Process jobs based on priority
        jobs_processed = 0
        processing_capacity = (1.0 - self.server_load) * self.job_processing_rate
        
        # Sort jobs by priority (higher priority first)
        self.job_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        # Process jobs matching the selected priority level
        for job in self.job_queue[:]:
            if job['priority'] == self.current_priority:
                if processing_capacity > 0:
                    job_size = job['size']
                    processing_time = job_size / (processing_capacity * 100)
                    
                    if processing_time <= 1.0:  # Can process in this step
                        self.job_queue.remove(job)
                        jobs_processed += 1
                        processing_capacity -= job_size / 100
        
        self.queue_length = len(self.job_queue)
        
        # Update server load (increases with queue, decreases with processing)
        load_change = (self.queue_length * 0.01) - (jobs_processed * 0.05)
        self.server_load = np.clip(self.server_load + load_change, 0.0, 1.0)
        
        # Calculate reward based on latency and efficiency
        # Lower latency = higher reward
        latency_penalty = self.queue_length * 0.1  # Penalty for long queue
        load_penalty = abs(self.server_load - 0.5) * 0.2  # Penalty for extreme loads
        processing_reward = jobs_processed * 0.5  # Reward for processing jobs
        priority_match_bonus = 0.1 if jobs_processed > 0 else -0.05  # Bonus for correct priority
        
        reward = processing_reward - latency_penalty - load_penalty + priority_match_bonus
        
        # Check termination conditions
        terminated = self.episode_steps >= self.max_episode_steps
        truncated = self.server_load >= 0.95 or self.queue_length >= self.max_queue_length
        
        observation = self._get_obs()
        info = self._get_info()
        info['jobs_processed'] = jobs_processed
        info['reward_components'] = {
            'processing_reward': processing_reward,
            'latency_penalty': -latency_penalty,
            'load_penalty': -load_penalty,
            'priority_bonus': priority_match_bonus
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional, for visualization)."""
        if self.render_mode == "human":
            print(f"Step: {self.episode_steps}")
            print(f"Server Load: {self.server_load:.2f}")
            print(f"Queue Length: {self.queue_length}")
            print(f"Current Priority: {self.current_priority}")
            print("-" * 40)

