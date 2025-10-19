import numpy as np

def ahp_weight(matrix):
    """
    计算AHP权重向量和一致性检验
    :param matrix: 成对比较矩阵 (numpy array)
    :return: 权重向量, CI, CR
    """
    n = matrix.shape[0]  # 矩阵阶数
    
    # 1. 计算特征向量和最大特征值
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue = np.max(eigenvalues.real)  # 取最大实部
    max_eigenvector = eigenvectors[:, np.argmax(eigenvalues.real)].real  # 取对应特征向量
    
    # 2. 归一化特征向量
    weights = max_eigenvector / np.sum(max_eigenvector)
    
    # 3. 计算一致性指标 CI
    CI = (max_eigenvalue - n) / (n - 1)
    
    # 4. 计算一致性比率 CR
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_dict.get(n, 1.45)  # 获取对应的RI值，默认为9阶的1.45
    CR = CI / RI if RI != 0 else 0  # 避免除以0
    
    return weights, CI, CR

# 示例：构造成对比较矩阵（3x3）
comparison_matrix = np.array([
    [1, 3, 5],
    [1/3, 1, 2],
    [1/5, 1/2, 1]
])

# 计算权重和一致性检验
weights, CI, CR = ahp_weight(comparison_matrix)

# 输出结果
print("权重向量:", weights)
print("一致性指标 CI:", CI)
print("一致性比率 CR:", CR)

# 判断一致性
if CR < 0.1:
    print("一致性通过!")
else:
    print("一致性不通过，请调整判断矩阵!")
