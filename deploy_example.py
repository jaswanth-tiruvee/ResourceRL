"""
Example: How to use the trained RL model in production
This demonstrates how the trained model would be deployed and used continuously.
"""

import numpy as np
from stable_baselines3 import PPO
from resource_scheduler_env import ResourceSchedulerEnv


def deploy_model_example():
    """
    Example of how to deploy the trained model in a production system.
    This would run continuously, making scheduling decisions in real-time.
    """
    
    # Step 1: Load the trained model (done once at startup)
    print("Loading trained model...")
    try:
        # Try zip file first, then directory
        import os
        model_path = "./models/ppo_final_model.zip"
        if not os.path.exists(model_path):
            model_path = "./models/ppo_final_model"
        model = PPO.load(model_path)
        print("✓ Model loaded successfully")
    except (FileNotFoundError, IsADirectoryError) as e:
        print("✗ Model not found. Please train the model first:")
        print("  python train_agent.py --algorithm PPO --timesteps 100000")
        return
    
    # Step 2: Create environment (or connect to real system)
    # In production, this would be your actual system state
    env = ResourceSchedulerEnv()
    
    # Step 3: Production loop (runs continuously)
    print("\n" + "="*60)
    print("Production Deployment Simulation")
    print("="*60)
    print("(In real production, this would run 24/7)\n")
    
    obs, info = env.reset()
    
    for step in range(20):  # Simulate 20 decision cycles
        # Get current system state
        current_state = obs
        server_load = current_state[0]
        queue_length = int(info['queue_length'])
        
        # Step 4: Use RL agent to make decision (this is the "running" part)
        action, _ = model.predict(current_state, deterministic=True)
        # Handle both scalar and array actions
        if isinstance(action, np.ndarray):
            priority_level = action.item() if action.ndim == 0 else action[0]
        else:
            priority_level = action
        
        # Step 5: Apply the decision (in production, this would update your scheduler)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Display decision
        print(f"Step {step+1:2d}: Queue={queue_length:2d}, Load={server_load:.2f} → "
              f"Priority={int(priority_level)} → Reward={reward:.2f}")
        
        if done:
            obs, info = env.reset()
            print("  [Environment reset]")
    
    print("\n" + "="*60)
    print("In production, this loop would run continuously,")
    print("making scheduling decisions every few seconds/minutes.")
    print("="*60)


def production_service_example():
    """
    Example of how this would be structured as a production service.
    This is pseudo-code showing the architecture.
    """
    
    print("\n" + "="*60)
    print("Production Service Architecture Example")
    print("="*60)
    
    code_example = '''
# Production Scheduler Service (pseudo-code)

class ResourceSchedulerService:
    def __init__(self):
        # Load model once at startup
        self.model = PPO.load("./models/ppo_final_model")
        self.running = True
    
    def start(self):
        """Start the service (runs continuously)"""
        while self.running:
            # 1. Get current system state
            state = self.get_system_state()
            
            # 2. Get optimal action from RL agent
            action, _ = self.model.predict(state, deterministic=True)
            
            # 3. Apply scheduling decision
            self.apply_scheduling_decision(action)
            
            # 4. Wait for next cycle
            time.sleep(1)  # or event-driven
    
    def get_system_state(self):
        """Get current server/queue state from monitoring system"""
        return np.array([
            get_server_load(),
            get_queue_length() / MAX_QUEUE,
            get_avg_job_size() / MAX_JOB_SIZE,
            get_current_priority() / 4.0
        ])
    
    def apply_scheduling_decision(self, priority_level):
        """Update scheduler with new priority"""
        update_job_queue_priority(int(priority_level))

# Service would run as:
# - REST API endpoint
# - Microservice
# - Background daemon
# - Kubernetes deployment
'''
    
    print(code_example)
    print("="*60)


if __name__ == "__main__":
    print("ResourceRL Production Deployment Example")
    print("="*60)
    print("\nThis demonstrates how the trained model would be used")
    print("in a production system that runs continuously.\n")
    
    # Run deployment example
    deploy_model_example()
    
    # Show architecture example
    production_service_example()
    
    print("\n" + "="*60)
    print("Key Takeaway:")
    print("  - Training: Offline, periodic (this app)")
    print("  - Deployment: Online, continuous (production service)")
    print("="*60)

