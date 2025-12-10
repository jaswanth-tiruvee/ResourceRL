"""
Example script demonstrating basic usage of the ResourceRL environment
"""

import numpy as np
from resource_scheduler_env import ResourceSchedulerEnv


def example_random_agent():
    """Example: Run a random agent for a few episodes."""
    print("=" * 60)
    print("Example: Random Agent")
    print("=" * 60)
    
    env = ResourceSchedulerEnv()
    
    for episode in range(3):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Initial State: Server Load={obs[0]:.2f}, Queue Length={info['queue_length']}")
        
        while not done and steps < 50:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
        
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final Queue Length: {info['queue_length']}")
        print(f"  Final Server Load: {info['server_load']:.2f}")
    
    env.close()


def example_static_policy():
    """Example: Run a static policy (always use medium priority)."""
    print("\n" + "=" * 60)
    print("Example: Static Policy (Baseline)")
    print("=" * 60)
    
    env = ResourceSchedulerEnv()
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print(f"\nInitial State: Server Load={obs[0]:.2f}, Queue Length={info['queue_length']}")
    
    while not done and steps < 50:
        # Always use medium priority (action = 2)
        action = 2
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        
        if steps % 10 == 0:
            print(f"  Step {steps}: Reward={reward:.2f}, Queue={info['queue_length']}, Load={info['server_load']:.2f}")
    
    print(f"\nFinal Results:")
    print(f"  Total Steps: {steps}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final Queue Length: {info['queue_length']}")
    print(f"  Final Server Load: {info['server_load']:.2f}")
    
    env.close()


if __name__ == "__main__":
    print("ResourceRL Environment Examples")
    print("=" * 60)
    
    # Run examples
    example_random_agent()
    example_static_policy()
    
    print("\n" + "=" * 60)
    print("To train an RL agent, run: python train_agent.py")
    print("=" * 60)

