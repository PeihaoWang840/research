import numpy as np
import random

def DE(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction):
    # 初始化参数
    VarSize = (1, nVar)  # 决策变量矩阵大小
    beta_min = 0.2  # 变异因子下界
    beta_max = 0.8  # 变异因子上界
    pCR = 0.2       # 交叉概率
    
    # 初始化种群
    class Individual:
        def __init__(self):
            self.Position = np.array([])
            self.Cost = float('inf')
    
    pop = [Individual() for _ in range(nPop)]
    
    for i in range(nPop):
        pos = np.random.uniform(VarMin, VarMax, VarSize)
        pop[i].Position = pos
        pop[i].Cost = CostFunction(pos)
        
        if pop[i].Cost < pop[0].Cost:
            best_sol = pop[i]
    
    BestCost = np.zeros(MaxIt)
    
    # 主循环
    for it in range(MaxIt):
        for i in range(nPop):
            x = pop[i].Position.copy()
            
            # 随机选择不同的个体a, b, c
            idx = list(range(nPop))
            idx.remove(i)
            a, b, c = random.sample(idx, 3)
            
            # 变异
            beta = np.random.uniform(beta_min, beta_max, VarSize)
            y = pop[a].Position + beta * (pop[b].Position - pop[c].Position)
            y = np.clip(y, VarMin, VarMax)  # 约束到范围
            
            # 交叉
            z = x.copy()
            j0 = random.randint(0, nVar-1)
            
            for j in range(nVar):
                if j == j0 or random.random() <= pCR:
                    z[j] = y[j]
            
            new_sol = Individual()
            new_sol.Position = z
            new_sol.Cost = CostFunction(z)
            
            if new_sol.Cost < pop[i].Cost:
                pop[i] = new_sol
                
                if new_sol.Cost < best_sol.Cost:
                    best_sol = new_sol
        
        BestCost[it] = best_sol.Cost
    
    bestf = BestCost[-1]
    bestx = best_sol.Position
    return bestf, bestx, BestCost

# 示例用法：
# import numpy as np
# def cost_func(x):
#     return np.sum(x**2)
# bestf, bestx, BestCost = DE(30, 100, 0, 1, 5, cost_func)