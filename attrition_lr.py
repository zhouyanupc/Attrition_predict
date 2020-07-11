'''
离职员工预测
'''
# 数据预处理
import pandas as pd
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
print(train['Attrition'].value_counts())

train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)
# 查看数据是否有空值
# print(train.isna().sum())

# 去掉无用的列 : 员工编码 标准工时
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'],axis=1)

# label encoder
from sklearn.preprocessing import LabelEncoder
attr = ['Age', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18','OverTime']

for feature in attr:
    label = LabelEncoder()
    train[feature] = label.fit_transform(train[feature])
    test[feature] = label.transform(test[feature])

train.to_csv('train_label_encoder.csv')

# 训练 模型:LR
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition', axis=1), train['Attrition'], test_size=0.2, random_state=33)

model = LogisticRegression(max_iter=1000,
                           verbose=True,
                           random_state=33,
                           tol=1e-4
                           )
model.fit(X_train, y_train)
predict = model.predict_proba(test)[:,1]
test['Attrition'] = predict

print(test['Attrition'])
test[['Attrition']].to_csv('submit_lr.csv')
print('submit_lr.csv saved')
# 转为二分类输出
# test['Attrition'] = test['Attrition'].map(lambda x:1 if x >= 0.5 else 0)
# test[['Attrition']].to_csv('submit_lr.csv')