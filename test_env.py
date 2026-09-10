import gymnasium as gym
env = gym.make("LunarLander-v3")
obs, _ = env.reset(seed=42)
print(f"Env OK: obs shape={obs.shape}, action_space={env.action_space}")
env.close()
print("Test passed!")
