import numpy as np
def pso(noP, Max_iteration, lb, ub, dim, fobj):
    # 定义PSO参数
    c1 = 2
    c2 = 2
    w_max = 0.9
    w_min = 0.2
    
    # 计算速度的最大值和最小值
    v_max = (ub - lb) * 0.2
    v_min = -v_max
    
    # 初始化变量
    iter_count = Max_iteration
    num_particles = noP
    dimensions = dim
    
    # 粒子的位置和速度矩阵的初始化
    pos = np.zeros((num_particles, dimensions))
    vel = np.zeros((num_particles, dimensions))
    
    # 每个粒子的最佳位置和最佳适应度值
    p_best_pos = np.zeros((num_particles, dimensions))
    p_best_score = np.full(num_particles, float('inf'))
    
    # 全局最佳位置和最佳适应度值
    g_best_pos = np.zeros(dimensions)
    g_best_score = float('inf')
    
    # 记录每代的全局最佳适应度值曲线
    cg_curve = np.zeros(iter_count)
        # 初始化粒子的位置
    for i in range(num_particles):
        pos[i] = lb + (ub - lb) * np.random.rand(dimensions)
    
    # 初始化粒子的速度
    vel = lb + (ub - lb) * 0.2 * np.random.rand(num_particles, dimensions)
    for t in range(iter_count):
        # 边界检查与约束处理
        pos = np.clip(pos, lb, ub)
        
        # 计算每个粒子的适应度值
        for i in range(num_particles):
            fitness = fobj(pos[i])
            
            if fitness < p_best_score[i]:
                p_best_score[i] = fitness
                p_best_pos[i] = pos[i]
                
            if fitness < g_best_score:
                g_best_score = fitness
                g_best_pos = pos[i]
        
        # 更新权重w
        w = w_max - (w_max - w_min) * t / iter_count
        
        # 更新速度和位置
        for i in range(num_particles):
            for j in range(dimensions):
                vel[i][j] = w * vel[i][j] + c1 * np.random.rand() * (p_best_pos[i][j] - pos[i][j]) \
                            + c2 * np.random.rand() * (g_best_pos[j] - pos[i][j])
                
                # 限制速度在[-v_max, v_max]范围内
                if vel[i][j] > v_max[j]:
                    vel[i][j] = v_max[j]
                elif vel[i][j] < v_min[j]:
                    vel[i][j] = v_min[j]
                    
            pos[i] += vel[i]
        
        # 记录全局最佳适应度值
        cg_curve[t] = g_best_score
    
    return g_best_score, g_best_pos, cg_curve
    # 示例目标函数（例如，求解x^2 + y^2的最小值）
def fobj(x):
    return np.sum(x**2)

# 参数设置
noP = 30         # 粒子数量
Max_iteration = 100   # 最大迭代次数
lb = [-3, -3]        # 决策变量下界
ub = [3, 3]          # 决策变量上界
dim = 2              # 维度

# 调用PSO函数
g_best_score, g_best_pos, cg_curve = pso(noP, Max_iteration, lb, ub, dim, fobj)

# 输出结果
print("全局最佳适应度值:", g_best_score)
print("全局最佳位置:", g_best_pos)