import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.efficient_kan.kan import KAN

class LunarLanderKAN(nn.Module):
    def __init__(self):
        super(LunarLanderKAN, self).__init__()
        self.kan = KAN([8, 128, 64, 4])

    def forward(self, x):
        return self.kan(x)

# 初始化环境和模型
env = gym.make('LunarLander-v2')#,render_mode='human')
model = LunarLanderKAN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# 尝试加载已保存的模型
checkpoint_path = 'kan_model_episode_test.pth'
csv_path = 'training_data.csv'
start_episode = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode'] + 1
    print(f"Loaded model from {checkpoint_path}, starting from episode {start_episode}")

# 训练模型
num_episodes = 100000
gamma = 0.99
rewards = []
losses = []
avg_rewards = []
avg_losses = []
plot_path = 'training_plot-avg.png'  # 保存图像的路径

def update_plot(avg_rewards, avg_losses, plot_path):
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards, label='Average Reward (per 100 episodes)')
    plt.xlabel('Episode (x100)')
    plt.ylabel('Value')
    plt.title('Average Reward')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

def update_csv(avg_rewards, avg_losses, csv_path):
    data = {'Average Reward': avg_rewards, 'Average Loss': avg_losses}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index_label='Episode (x100)')

for episode in range(start_episode, num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # if episode % 100 == 0:
        #     env.render()

        if isinstance(state, (list, tuple)) and len(state) != 8:
            print(f"Warning: Unexpected state length {len(state)}")
            state = np.zeros(8)

        state_tensor = torch.FloatTensor(np.array(state)).to(device).unsqueeze(0)
        action_probs = model(state_tensor)
        action = torch.argmax(action_probs).item()

        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, done, truncated, _ = step_result
        elif len(step_result) == 4:
            next_state, reward, done, _ = step_result
        else:
            print(f"Warning: Unexpected step result length {len(step_result)}")
            next_state, reward, done, _ = (np.zeros(8), 0, True, {})

        total_reward += reward

        if isinstance(next_state, (list, tuple)) and len(next_state) != 8:
            print(f"Warning: Unexpected next_state length {len(next_state)}")
            next_state = np.zeros(8)

        next_state_tensor = torch.FloatTensor(np.array(next_state)).to(device).unsqueeze(0)
        target = reward + gamma * torch.max(model(next_state_tensor)).item() * (1 - done)

        target_tensor = action_probs.clone()
        target_tensor[0, action] = target

        loss = criterion(action_probs, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    rewards.append(total_reward)
    losses.append(loss.item())

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        avg_loss = np.mean(losses[-100:])
        avg_rewards.append(avg_reward)
        avg_losses.append(avg_loss)
        print(f"Episode {episode + 1}, Average Reward: {avg_reward}, Average Loss: {avg_loss}")
        torch.save({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        update_plot(avg_rewards, avg_losses, plot_path)
        update_csv(avg_rewards, avg_losses, csv_path)

# 最后更新一次图像和 CSV 文件
update_plot(avg_rewards,  plot_path)
update_csv(avg_rewards, avg_losses, csv_path)

# 测试模型
for episode in range(0, num_episodes, 100):  # 每100轮测试一次
    state, _ = env.reset()
    done = False

    while not done:
        # env.render()
        if isinstance(state, (list, tuple)) and len(state) != 8:
            print(f"Warning: Unexpected state length {len(state)}")
            state = np.zeros(8)

        state_tensor = torch.FloatTensor(np.array(state)).to(device).unsqueeze(0)
        action_probs = model(state_tensor)
        action = torch.argmax(action_probs).item()
        step_result = env.step(action)

        if len(step_result) == 5:
            next_state, _, done, truncated, _ = step_result
        elif len(step_result) == 4:
            next_state, _, done, _ = step_result
        else:
            print(f"Warning: Unexpected step result length {len(step_result)}")
            next_state, _, done, _ = (np.zeros(8), 0, True, {})

        state = next_state

env.close()
