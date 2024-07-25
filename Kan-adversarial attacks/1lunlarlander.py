import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.efficient_kan.kan import KAN

# 创建 Lunar Lander 环境
env = gym.make('LunarLander-v2', render_mode="human")

# 定义 KAN 模型
class LunarLanderKAN(nn.Module):
    def __init__(self):
        super(LunarLanderKAN, self).__init__()
        self.kan = KAN([8, 64, 4])  # 输入是8维状态，输出是4个动作

    def forward(self, x):
        return self.kan(x)

model = LunarLanderKAN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# 训练模型
num_episodes = 1000
gamma = 0.99  # 折扣因子

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0


    while not done:
        # 打印状态以调试
       # print(f"Initial state: {state}")

        if isinstance(state, (list, tuple)) and len(state) != 8:
            print(f"Warning: Unexpected state length {len(state)}")
            state = np.zeros(8)  # fallback to zeros if state is invalid

        state_tensor = torch.FloatTensor(np.array(state)).to(device).unsqueeze(0)
        action_probs = model(state_tensor)
        action = torch.argmax(action_probs).item()

        step_result = env.step(action)
        print(f"Step result: {step_result}")

        if len(step_result) != 4:
            print(f"Warning: Unexpected step result length {len(step_result)}")
            next_state, reward, done, _ = (np.zeros(8), 0, True, {})
        else:
            next_state, reward, done, _ = step_result

        total_reward += reward

        if isinstance(next_state, (list, tuple)) and len(next_state) != 8:
            print(f"Warning: Unexpected next_state length {len(next_state)}")
            next_state = np.zeros(8)  # fallback to zeros if next_state is invalid

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

# 测试模型
state = env.reset()
done = False

while not done:
    if episode % 100 == 0:
        env.render()
    print(f"Initial state: {state}")
    if isinstance(state, (list, tuple)) and len(state) != 8:
        print(f"Warning: Unexpected state length {len(state)}")
        state = np.zeros(8)  # fallback to zeros if state is invalid

    state_tensor = torch.FloatTensor(np.array(state)).to(device).unsqueeze(0)
    action_probs = model(state_tensor)
    action = torch.argmax(action_probs).item()
    step_result = env.step(action)

    print(f"Step result: {step_result}")
    if len(step_result) != 4:
        print(f"Warning: Unexpected step result length {len(step_result)}")
        next_state, _, done, _ = (np.zeros(8), 0, True, {})
    else:
        next_state, _, done, _ = step_result

    state = next_state

env.close()
