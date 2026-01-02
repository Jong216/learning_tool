'''
GBDT.CART 的 Docstring
使用sklearn包工具实现官方糖尿病患者的数据库预测任务
'''
# 1.构建数据
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
datasets = load_diabetes()

X = datasets.data
Y = datasets.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,        # 测试集比例 30%
    train_size=None,      # 训练集比例自动计算
    random_state=42,      # 随机种子，确保可重复性
    shuffle=True,         # 打乱数据
    stratify=None,        # 回归任务通常不进行分层
)

print(f"训练集: X_train={X_train.shape}, Y_train={Y_train.shape}")
print(f"测试集: X_test={X_test.shape}, Y_test={Y_test.shape}")

# 2.建树
from sklearn.tree import DecisionTreeRegressor
cart = DecisionTreeRegressor(
    max_depth=5,           # 限制最大深度
    min_samples_split=10,  # 节点分裂所需的最小样本数
    min_samples_leaf=5     # 叶节点的最小样本数
)

# 3.训练回归树
cart = cart.fit(X_train, Y_train)

# 4.评估
score = cart.score(X_test, Y_test)
print(f"最终得分：{score}")


# 绘制预测结果与实际值散点图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置全局字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

import numpy as np
 
preds = cart.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制实际值 vs 预测值的散点图
ax.scatter(Y_test, preds, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)

# 绘制理想预测线 (y = x)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='理想预测 (y = x)')

# 设置图表属性
ax.set_xlabel('实际值 (Y_test)', fontsize=12)
ax.set_ylabel('预测值 (preds)', fontsize=12)
ax.set_title(f'糖尿病进展预测结果 (R^2 = {score:.3f})', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

# 添加一些统计信息到图表
ax.text(0.05, 0.95, f'样本数: {len(Y_test)}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# 打印预测结果的统计信息
print("\n预测结果统计:")
print(f"预测值范围: [{preds.min():.2f}, {preds.max():.2f}]")
print(f"实际值范围: [{Y_test.min():.2f}, {Y_test.max():.2f}]")
print(f"预测均值: {preds.mean():.2f}")
print(f"实际均值: {Y_test.mean():.2f}")

# 计算并显示额外的评估指标
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(Y_test, preds)
mse = mean_squared_error(Y_test, preds)
rmse = np.sqrt(mse)

print(f"\n额外评估指标:")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")


    
 