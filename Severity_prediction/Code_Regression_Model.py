import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn import tree
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  
from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBRegressor

df = pd.read_csv('Fe_ALL.csv')
data = df._values

x = data[:,0:39]
y = data[:,41]

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.25)


# 建立贝叶斯岭回归模型
br_model = BayesianRidge()

# 普通线性回归
lr_model = LinearRegression()

# 弹性网络回归模型
etc_model = ElasticNet()

# 支持向量机回归
svr_model = SVR()

# 梯度增强回归模型对象
gbr_model = GradientBoostingRegressor()

Cart_model = tree.DecisionTreeRegressor(max_depth=6,
                                 random_state=40,
                                 splitter="random",
                                 min_samples_split=4)

XG_model = XGBRegressor(max_depth = 6,
                        n_estimators = 200,
                        subsample=0.7,
                        reg_lambda=0.3,
                        reg_alpha = 0.6,
                        min_child_weight = 4,
                        learning_rate = 0.02,
                        scale_pos_weight=2.5)
# 不同模型的名称列表
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR','CART','XGboost']
# 不同回归模型
model_dic = [br_model, lr_model, etc_model, svr_model, gbr_model,Cart_model,XG_model]
# 交叉验证结果
cv_score_list = []
# 各个回归模型预测的y值列表
pre_y_list = []
pre_ytrain_list = []

n_folds = 5
# 读出每个回归模型对象
for model in model_dic:
    # 将每个回归模型导入交叉检验
    scores = cross_val_score(model, x_train, y_train, cv=n_folds)
    # 将交叉检验结果存入结果列表
    cv_score_list.append(scores)
    # 将回归训练中得到的预测y存入列表
    pre_y_list.append(model.fit(x_train, y_train).predict(x_test))
    pre_ytrain_list.append(model.fit(x_train, y_train).predict(x_train))
### 模型效果指标评估 ###
# 获取样本量，特征数
# n_sample, n_feature = x.shape
# 回归评估指标对象列表
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error]
# 回归评估指标列表
model_metrics_list = []
# 循环每个模型的预测结果
for i in range(7):
    pre_y_train = pre_ytrain_list[i]
    pre_y = pre_y_list[i]
    y_pre_all = np.hstack((pre_y_train,pre_y))
    y_all = np.hstack((y_train,y_test))
    tmp_list = []
    # 循环每个指标对象
    for mdl in model_metrics_name:
        # 计算每个回归指标结果
        tmp_score = mdl(y_test, pre_y)
        # 将结果存入临时列表
        tmp_list.append(tmp_score)
    # 将结果存入回归评估列表
    model_metrics_list.append(tmp_list)
df_score = pd.DataFrame(cv_score_list, index=model_names)

# 各个交叉验证的结果
print(df_score)

