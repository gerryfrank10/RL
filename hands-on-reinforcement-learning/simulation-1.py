import gymnasium as gym
env = gym.make('CarRacing-v3', render_mode='human')
observation, info = env.reset(seed=42)
done = False
while not done:
    action = env.action_space.sample()  # Replace with your action logic
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()