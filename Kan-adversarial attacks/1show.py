import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.efficient_kan.kan import KAN
class LunarLanderKAN(nn.Module):
    def __init__(self):
        super(LunarLanderKAN, self).__init__()
        self.kan = KAN([8, 64,64, 4])

    def forward(self, x):
        return self.kan(x)

# 初始化环境和模型
env = gym.make('LunarLander-v2', render_mode="human")  # 使用 render_mode="human" 来渲染动画
model = LunarLanderKAN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载保存的模型权重
model.load_state_dict(torch.load('kan_model_episode_5900.pth'))  # 修改为你实际的模型文件路径
model.eval()  # 设置模型为评估模式

# 演示动画
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    env.render()  # 渲染环境

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

    state = next_state

print(f"Total Reward: {total_reward}")
env.close()
