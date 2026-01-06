'''
GBDT.gbdt 的 Docstring
使用GBDT对官方数据集作回归分析
'''
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小

# url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
# urllib.request.urlretrieve(url, "BostonHousing.csv")

# 构建数据集
df = pd.read_csv("BostonHousing.csv")
print(df.head())

target_column = 'medv'  # 波士顿房价数据集的目标变量

X = df.drop(columns=[target_column])  # 所有特征列

y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    shuffle=True, random_state=42)

# 训练模型
params = {'n_estimators': 500, # 弱分类器的个数
          'max_depth': 3,       # 弱分类器（CART回归树）的最大深度
          'min_samples_split': 5, # 分裂内部节点所需的最小样本数
          'learning_rate': 0.05,  # 学习率
          'loss': 'ls'}   

gbdt_model = GradientBoostingRegressor(**params)

gbdt_model.fit(X_train, y_train)

# 模型预测
y_predict = gbdt_model.predict(X_test)

# 绘制折线图
y_test_reset = y_test.reset_index(drop=True)

# 使用统一的X轴（样本序号）

x_axis = np.arange(len(y_test_reset))

# 绘制真实值和预测值，使用相同的X轴

plt.plot(x_axis, y_test_reset, 'b-', label='真实房价', linewidth=1.5, marker='o', markersize=3)

plt.plot(x_axis, y_predict, 'r--', label='预测房价', linewidth=1.5, marker='x', markersize=3)

plt.xlabel('样本序号', fontsize=12)

plt.ylabel('房价 (千美元)', fontsize=12)

plt.title('真实房价 vs 预测房价 (统一X轴)', fontsize=14)

plt.legend(loc='best', fontsize=10)

plt.grid(True, alpha=0.3)

plt.tight_layout()


plt.show()