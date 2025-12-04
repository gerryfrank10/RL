import gymnasium as gym

env = gym.make('BipedalWalker-v3', render_mode='human')
for episode in range(100):
    observation, info = env.reset(seed=42)
    done = False
    for t in range(1000):
        action = env.action_space.sample()  # Replace with your action logic
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()