import numpy as np
def simulated_annealing(Mmax, l, u, dim, fobj):
    # 初始化参数
    TolFun = 1e-10  # 容差值
    x0 = (u - l) * np.random.rand(1, dim) + l  # 初始位置
    x = x0.copy()
    fx = fobj(x)
    f0 = fx
    count = 1
    curve = []  # 用于存储每一步的最优函数值

    for m in range(1, Mmax + 1):
        T = m / Mmax  # 温度参数
        mu = 10 ** (T * 1000)
        
        for k in range(100):  # 每个温度下进行100次尝试
            # 计算步长
            y = 2 * np.random.rand(1, dim) - 1
            dx = mu_inv(y, mu) * (u - l)
            
            x1 = x + dx  # 新的位置
            
            # 边界检查，将超出边界的值限制在[l, u]范围内
            x1 = np.clip(x1, l, u)
            
            fx1 = fobj(x1)  # 计算新位置的函数值
            df = fx1 - fx  # 函数变化
            
            # 满足Metropolis准则，接受新的解
            if df < 0 or (np.random.rand() < np.exp(-T * df / (abs(fx) + np.finfo(float).eps)/ TolFun)):
                x, fx = x1, fx1
                
            # 更新全局最优解
            if fx1 < f0:
                x0, f0 = x1, fx1
                
        curve.append(f0)
        
    Best_pos = x0
    Best_score = f0
    
    return Best_score, Best_pos, np.array(curve)

def mu_inv(y, mu):
    return ((1 + mu) ** abs(y) - 1) / mu * np.sign(y)

import numpy as np

def fobj(x):
    return x[0]**2 + x[1]**2  # 目标函数

Mmax = 100
lb = np.array([0, 0])
ub = np.array([1, 1])
dim = 2

Best_score, Best_pos, curve = simulated_annealing(Mmax, lb, ub, dim, fobj)

print(f'Best score: {Best_score}')
print(f'Best position: {Best_pos}')
import matplotlib.pyplot as plt
plt.plot(curve)
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.show()