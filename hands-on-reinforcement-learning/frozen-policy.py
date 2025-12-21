import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random


# Improved grid generation with guaranteed solvability
def generate_frozen_lake_grid(size=16, hole_prob=0.3):
    while True:
        grid = []
        for i in range(size):
            row = []
            for j in range(size):
                if (i == 0 and j == 0):
                    row.append('S')
                elif (i == size - 1 and j == size - 1):
                    row.append('G')
                else:
                    row.append('H' if random.random() < hole_prob else 'F')
            grid.append(row)

        # Check if path exists using BFS
        if is_solvable(grid, size):
            return grid


def is_solvable(grid, size):
    """Check if there's a path from S to G"""
    visited = set()
    queue = [(0, 0)]

    while queue:
        i, j = queue.pop(0)
        if (i, j) in visited or i < 0 or i >= size or j < 0 or j >= size:
            continue
        if grid[i][j] == 'H':
            continue
        if grid[i][j] == 'G':
            return True

        visited.add((i, j))
        queue.extend([(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])

    return False


# Improved DQN with dueling architecture
class DuelingDQN(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


# Enhanced anti-loop wrapper with position tracking
class EnhancedAntiLoopWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.5, max_revisits=3):
        super().__init__(env)
        self.penalty = penalty
        self.max_revisits = max_revisits
        self.visit_count = None
        self.step_count = 0
        self.max_steps = 200

    def reset(self, **kwargs):
        self.visit_count = {}
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Track visits
        if next_state in self.visit_count:
            self.visit_count[next_state] += 1
            # Increasing penalty for revisits
            reward -= self.penalty * self.visit_count[next_state]
        else:
            self.visit_count[next_state] = 1

        # Add small step penalty to encourage efficiency
        reward -= 0.01

        # Truncate if too many steps
        if self.step_count >= self.max_steps:
            truncated = True

        return next_state, reward, terminated, truncated, info


def one_hot(state, size):
    vec = np.zeros(size)
    vec[state] = 1
    return vec


# Training function with improvements
def train_dqn(env, num_episodes=15000, use_prioritized=True):
    state_space = env.observation_space.n
    action_space = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hyperparameters
    gamma = 0.99
    lr = 5e-4
    batch_size = 128
    epsilon = 1.0
    epsilon_decay = 0.9998
    min_epsilon = 0.01
    target_update_freq = 500
    warmup_episodes = 1000

    # Networks
    policy_net = DuelingDQN(state_space, action_space).to(device)
    target_net = DuelingDQN(state_space, action_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Replay buffer
    if use_prioritized:
        replay_buffer = PrioritizedReplayBuffer(capacity=50_000)
        beta = 0.4
        beta_increment = 0.001
    else:
        replay_buffer = deque(maxlen=50_000)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = deque(maxlen=100)

    step_count = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            step_count += 1
            episode_length += 1

            # Epsilon-greedy action selection
            if episode < warmup_episodes or np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                s = torch.tensor(one_hot(state, state_space), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = policy_net(s).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store transition
            if use_prioritized:
                replay_buffer.push(one_hot(state, state_space), action, reward,
                                   one_hot(next_state, state_space), done)
            else:
                replay_buffer.append((one_hot(state, state_space), action, reward,
                                      one_hot(next_state, state_space), done))

            state = next_state

            # Training step
            if len(replay_buffer) >= batch_size and episode >= warmup_episodes:
                if use_prioritized:
                    states, actions, rewards, next_states, dones, indices, weights = \
                        replay_buffer.sample(batch_size, beta)
                    weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
                    beta = min(1.0, beta + beta_increment)
                else:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
                    weights = torch.ones(batch_size, 1).to(device)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                # Double DQN
                q_values = policy_net(states).gather(1, actions)

                with torch.no_grad():
                    # Use policy net to select action
                    next_actions = policy_net(next_states).argmax(1, keepdim=True)
                    # Use target net to evaluate action
                    next_q_values = target_net(next_states).gather(1, next_actions)
                    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                # Prioritized experience replay
                td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
                loss = (weights * (q_values - target_q_values) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()

                if use_prioritized:
                    replay_buffer.update_priorities(indices, td_errors.flatten() + 1e-6)

            # Update target network
            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_rate.append(1 if episode_reward > 0 else 0)

        # Logging
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            success = np.mean(success_rate) * 100
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Success Rate: {success:.1f}%")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Buffer Size: {len(replay_buffer)}")

    return policy_net, episode_rewards, success_rate

def play_policy(env, policy, filename='frozenlake-qlearning-policy.mp4', fps=2):
    frames = []
    state, info = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        frame = env.render()
        frames.append(frame)
        action = int(policy[state])
        state, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    imageio.mimsave(filename, frames, fps=fps)

# Example usage
if __name__ == "__main__":
    # Generate grid
    # grid = generate_frozen_lake_grid(size=8, hole_prob=0.2)
    # desc = [''.join(row) for row in grid]

    grid = [['S', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'H', 'F', 'H', 'F', 'F', 'F'], ['F', 'H', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'F'], ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'F', 'F'], ['F', 'F', 'F', 'F', 'F', 'H', 'H', 'F', 'H', 'F', 'F', 'H', 'F', 'H', 'F', 'F'], ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H'], ['H', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F'], ['F', 'F', 'H', 'F', 'H', 'F', 'F', 'H', 'F', 'H', 'F', 'H', 'F', 'F', 'F', 'F'], ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'H', 'H', 'F', 'F'], ['F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H'], ['F', 'F', 'F', 'F', 'H', 'H', 'H', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'F'], ['F', 'F', 'F', 'F', 'H', 'H', 'H', 'F', 'F', 'H', 'F', 'F', 'H', 'F', 'H', 'F'], ['F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F'], ['F', 'F', 'H', 'F', 'F', 'H', 'F', 'F', 'F', 'H', 'H', 'H', 'F', 'F', 'F', 'H'], ['F', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'], ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'H', 'F', 'H', 'F', 'F', 'F'], ['F', 'F', 'H', 'H', 'F', 'F', 'F', 'H', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'G']]


    # Create environment
    env = gym.make('FrozenLake-v1', desc=grid, is_slippery=False, render_mode='rgb_array')
    env = EnhancedAntiLoopWrapper(env)

    # Train
    print("Training DQN agent...")
    policy_net, rewards, success = train_dqn(env, num_episodes=10000)

    print(f"\nFinal Success Rate: {np.mean(list(success)) * 100:.1f}%")

    # Extract policy
    state_space = env.observation_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dqn_policy = np.zeros(state_space, dtype=int)
    for s in range(state_space):
        state_vec = torch.tensor(one_hot(s, state_space), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            dqn_policy[s] = policy_net(state_vec).argmax().item()

    # Save  policy
    np.save('dqn_policy.npy', dqn_policy)
    # Play and save video of the learned policy
    play_policy(env, dqn_policy, filename='frozenlake-dqn-policy-4.mp4', fps=2)
    print("\nPolicy extracted successfully!")