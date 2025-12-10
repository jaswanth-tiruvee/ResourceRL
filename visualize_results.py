"""
Visualization script for RL training results
Plots cumulative reward curves and performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
import os
import glob


def plot_training_curves(log_dir='./models/logs', save_path='./models/training_curves.png'):
    """
    Plot cumulative reward curves from training logs.
    
    Args:
        log_dir: Directory containing training logs
        save_path: Path to save the plot
    """
    # Try to load training data
    try:
        df = load_results(log_dir)
    except Exception as e:
        print(f"Could not load training logs: {e}")
        print("Skipping training curves plot. Use TensorBoard to view training progress.")
        return
    
    if df.empty:
        print("No training data found. Please train the agent first.")
        return
    
    # Calculate cumulative rewards
    df['cumulative_reward'] = df['r'].cumsum()
    df['episode'] = range(len(df))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL Agent Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Cumulative Reward over Episodes
    axes[0, 0].plot(df['episode'], df['cumulative_reward'], linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Cumulative Reward', fontsize=12)
    axes[0, 0].set_title('Cumulative Reward Curve', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Episode Rewards (with moving average)
    window_size = max(10, len(df) // 20)
    df['reward_ma'] = df['r'].rolling(window=window_size, center=True).mean()
    
    axes[0, 1].plot(df['episode'], df['r'], alpha=0.3, color='#A23B72', label='Raw')
    axes[0, 1].plot(df['episode'], df['reward_ma'], linewidth=2, color='#F18F01', label=f'Moving Avg ({window_size})')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Episode Reward', fontsize=12)
    axes[0, 1].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Episode Length
    if 'l' in df.columns:
        axes[1, 0].plot(df['episode'], df['l'], linewidth=2, color='#C73E1D')
        axes[1, 0].set_xlabel('Episode', fontsize=12)
        axes[1, 0].set_ylabel('Episode Length', fontsize=12)
        axes[1, 0].set_title('Episode Length', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution
    axes[1, 1].hist(df['r'], bins=30, color='#6A994E', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(df['r'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["r"].mean():.2f}')
    axes[1, 1].set_xlabel('Reward', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_evaluation_comparison(results_path='./models/evaluation_results.csv', 
                               save_path='./models/evaluation_comparison.png'):
    """
    Plot comparison between RL agent and baseline.
    
    Args:
        results_path: Path to evaluation results CSV
        save_path: Path to save the plot
    """
    if not os.path.exists(results_path):
        print(f"Evaluation results not found at {results_path}")
        print("Please run training with evaluation first.")
        return
    
    df = pd.read_csv(results_path)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('RL Agent vs Baseline Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Bar chart comparison
    metrics = df['Metric'].values
    rl_values = df['RL Agent'].values
    baseline_values = df['Baseline'].values
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, rl_values, width, label='RL Agent', color='#2E86AB', alpha=0.8)
    axes[0].bar(x + width/2, baseline_values, width, label='Baseline', color='#A23B72', alpha=0.8)
    
    axes[0].set_xlabel('Metric', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Improvement percentage
    improvements = []
    improvement_labels = []
    
    for i, metric in enumerate(metrics):
        if baseline_values[i] != 0:
            if 'Queue Length' in metric:
                # For queue length, lower is better
                improvement = ((baseline_values[i] - rl_values[i]) / baseline_values[i]) * 100
            else:
                # For rewards, higher is better
                improvement = ((rl_values[i] - baseline_values[i]) / abs(baseline_values[i])) * 100
            improvements.append(improvement)
            improvement_labels.append(metric.replace('Mean ', ''))
    
    colors = ['#6A994E' if x > 0 else '#C73E1D' for x in improvements]
    axes[1].barh(improvement_labels, improvements, color=colors, alpha=0.8)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_xlabel('Improvement (%)', fontsize=12)
    axes[1].set_title('Performance Improvement', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (label, value) in enumerate(zip(improvement_labels, improvements)):
        axes[1].text(value + (1 if value > 0 else -1), i, f'{value:.1f}%', 
                    va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation comparison saved to {save_path}")
    plt.close()


def plot_episode_rewards_comparison(rl_rewards, baseline_rewards, 
                                    save_path='./models/episode_rewards_comparison.png'):
    """
    Plot episode-by-episode reward comparison.
    
    Args:
        rl_rewards: List of episode rewards from RL agent
        baseline_rewards: List of episode rewards from baseline
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Episode Rewards: RL Agent vs Baseline', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(rl_rewards) + 1)
    
    # Plot 1: Line plot
    axes[0].plot(episodes, rl_rewards, marker='o', label='RL Agent', 
                color='#2E86AB', linewidth=2, markersize=6, alpha=0.7)
    axes[0].plot(episodes, baseline_rewards, marker='s', label='Baseline', 
                color='#A23B72', linewidth=2, markersize=6, alpha=0.7)
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Episode Reward', fontsize=12)
    axes[0].set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    data = [rl_rewards, baseline_rewards]
    bp = axes[1].boxplot(data, labels=['RL Agent', 'Baseline'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)
    
    axes[1].set_ylabel('Episode Reward', fontsize=12)
    axes[1].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Episode rewards comparison saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize RL training results')
    parser.add_argument('--log-dir', type=str, default='./models/logs',
                       help='Directory containing training logs')
    parser.add_argument('--results-file', type=str, default='./models/evaluation_results.csv',
                       help='Path to evaluation results CSV')
    
    args = parser.parse_args()
    
    # Plot training curves
    print("Generating training curves...")
    plot_training_curves(log_dir=args.log_dir)
    
    # Plot evaluation comparison
    print("\nGenerating evaluation comparison...")
    plot_evaluation_comparison(results_path=args.results_file)
    
    print("\nVisualization complete!")

