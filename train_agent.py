"""
Training script for Resource Scheduling RL Agent
Uses Stable Baselines3 with PPO and A2C algorithms
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from resource_scheduler_env import ResourceSchedulerEnv


def create_env():
    """Create and wrap the environment."""
    env = ResourceSchedulerEnv()
    env = Monitor(env)
    return env


def train_agent(algorithm='PPO', total_timesteps=100000, save_dir='./models'):
    """
    Train the RL agent using specified algorithm.
    
    Args:
        algorithm: 'PPO' or 'A2C'
        total_timesteps: Number of training timesteps
        save_dir: Directory to save models and logs
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/logs', exist_ok=True)
    os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([create_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([create_env])
    
    # Initialize the agent
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=f'{save_dir}/logs'
        )
    elif algorithm == 'A2C':
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            tensorboard_log=f'{save_dir}/logs'
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{save_dir}/best_model',
        log_path=f'{save_dir}/logs',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f'{save_dir}/checkpoints',
        name_prefix='rl_model'
    )
    
    print(f"Training {algorithm} agent for {total_timesteps} timesteps...")
    print(f"Model will be saved to {save_dir}")
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f'{save_dir}/{algorithm.lower()}_final_model')
    print(f"Training complete! Model saved to {save_dir}/{algorithm.lower()}_final_model")
    
    return model, env


def evaluate_agent(model, env, num_episodes=10):
    """
    Evaluate the trained agent.
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    queue_lengths = []
    server_loads = []
    
    for episode in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = [{}]
        
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            # Handle vectorized environment (returns 4 values: obs, rewards, dones, infos)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                done = done[0] if isinstance(done, (np.ndarray, list)) else done
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated[0] if isinstance(terminated, np.ndarray) else terminated
                done = done or (truncated[0] if isinstance(truncated, np.ndarray) else truncated)
            
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1
            
            if info and len(info) > 0:
                episode_info = info[0] if isinstance(info, list) else info
                if isinstance(episode_info, dict):
                    if 'queue_length' in episode_info:
                        queue_lengths.append(episode_info['queue_length'])
                    if 'server_load' in episode_info:
                        server_loads.append(episode_info['server_load'])
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
        'mean_server_load': np.mean(server_loads) if server_loads else 0,
        'episode_rewards': episode_rewards
    }
    
    return results


def compare_with_baseline(env, num_episodes=10):
    """
    Compare RL agent performance with baseline static scheduling (FIFO).
    
    Returns:
        Dictionary with baseline metrics
    """
    episode_rewards = []
    queue_lengths = []
    server_loads = []
    
    for episode in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = [{}]
        
        done = False
        episode_reward = 0
        
        while not done:
            # Baseline: Always use medium priority (action = 2)
            action = np.array([2])
            step_result = env.step(action)
            
            # Handle vectorized environment (returns 4 values: obs, rewards, dones, infos)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                done = done[0] if isinstance(done, (np.ndarray, list)) else done
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated[0] if isinstance(terminated, np.ndarray) else terminated
                done = done or (truncated[0] if isinstance(truncated, np.ndarray) else truncated)
            
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
            if info and len(info) > 0:
                episode_info = info[0] if isinstance(info, list) else info
                if isinstance(episode_info, dict):
                    if 'queue_length' in episode_info:
                        queue_lengths.append(episode_info['queue_length'])
                    if 'server_load' in episode_info:
                        server_loads.append(episode_info['server_load'])
        
        episode_rewards.append(episode_reward)
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
        'mean_server_load': np.mean(server_loads) if server_loads else 0,
        'episode_rewards': episode_rewards
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL agent for resource scheduling')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'A2C'],
                       help='RL algorithm to use (PPO or A2C)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    # Train the agent
    model, env = train_agent(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        save_dir='./models'
    )
    
    # Evaluate the trained agent
    print("\n" + "="*50)
    print("Evaluating trained agent...")
    print("="*50)
    rl_results = evaluate_agent(model, env, num_episodes=args.eval_episodes)
    
    print(f"\nRL Agent Results ({args.algorithm}):")
    print(f"  Mean Reward: {rl_results['mean_reward']:.2f} ± {rl_results['std_reward']:.2f}")
    print(f"  Mean Queue Length: {rl_results['mean_queue_length']:.2f}")
    print(f"  Mean Server Load: {rl_results['mean_server_load']:.2f}")
    
    # Compare with baseline
    print("\n" + "="*50)
    print("Evaluating baseline (static scheduling)...")
    print("="*50)
    baseline_results = compare_with_baseline(env, num_episodes=args.eval_episodes)
    
    print(f"\nBaseline Results (Static Scheduling):")
    print(f"  Mean Reward: {baseline_results['mean_reward']:.2f} ± {baseline_results['std_reward']:.2f}")
    print(f"  Mean Queue Length: {baseline_results['mean_queue_length']:.2f}")
    print(f"  Mean Server Load: {baseline_results['mean_server_load']:.2f}")
    
    # Calculate improvement
    improvement = ((rl_results['mean_reward'] - baseline_results['mean_reward']) / 
                   abs(baseline_results['mean_reward'])) * 100
    
    latency_improvement = ((baseline_results['mean_queue_length'] - rl_results['mean_queue_length']) / 
                          baseline_results['mean_queue_length']) * 100
    
    print("\n" + "="*50)
    print("Performance Comparison:")
    print("="*50)
    print(f"  Reward Improvement: {improvement:.1f}%")
    print(f"  Latency Reduction (queue length): {latency_improvement:.1f}%")
    
    # Save results
    results_df = pd.DataFrame({
        'Metric': ['Mean Reward', 'Std Reward', 'Mean Queue Length', 'Mean Server Load'],
        'RL Agent': [
            rl_results['mean_reward'],
            rl_results['std_reward'],
            rl_results['mean_queue_length'],
            rl_results['mean_server_load']
        ],
        'Baseline': [
            baseline_results['mean_reward'],
            baseline_results['std_reward'],
            baseline_results['mean_queue_length'],
            baseline_results['mean_server_load']
        ]
    })
    
    results_df.to_csv('./models/evaluation_results.csv', index=False)
    print(f"\nResults saved to ./models/evaluation_results.csv")

