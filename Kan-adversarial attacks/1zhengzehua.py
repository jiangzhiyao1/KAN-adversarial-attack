import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.efficient_kan.kan import KAN



# LunarLanderKAN 类使用 KAN 模型
class LunarLanderKAN(nn.Module):
    def __init__(self):
        super(LunarLanderKAN, self).__init__()
        self.kan = KAN([8, 64,64, 4])

    def forward(self, x):
        return self.kan(x)

# 初始化环境和模型
env = gym.make('LunarLander-v2',render_mode='human')
model = LunarLanderKAN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# 训练模型
num_episodes = 100000
gamma = 0.99

for episode in range(num_episodes):
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

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss.item()}")
        torch.save(model.state_dict(), f'kan_model_episode_{episode}.pth')

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