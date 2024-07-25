import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 给定数据点
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 0, -1, 0])

# 构建三次样条函数
cs = CubicSpline(x, y)

# 绘制样条函数
x_new = np.linspace(0, 4, 100)
y_new = cs(x_new)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_new, y_new, label='Cubic Spline')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()
