import xlearn as xl
import pandas as pd
from sklearn.datasets import dump_svmlight_file

df = pd.read_csv("D:/1007studying/2021/DATA/ts_FE.csv")
df1 = pd.read_csv("D:/1007studying/2021/DATA/tr_FE.csv")
features = pd.read_csv('D:/1007studying/2021/DATA/feature.csv')
x_columns = features.head(30)['feature'].tolist() #特征重要性排序中，选择前30的特征

y = df.click  # y为数据的label值
dummy = pd.get_dummies(df[x_columns])
#mat = dummy.as_matrix()
mat = dummy.values

y1 = df1.click  # y为数据的label值
dummy = pd.get_dummies(df1[x_columns])
#mat = dummy.as_matrix()
mat1 = dummy.values

dump_svmlight_file(mat, y, 'test.libsvm', zero_based=False)
dump_svmlight_file(mat1, y1, 'train.libsvm', zero_based=False)


xfm = xl.create_fm()
xfm.setTrain("train.libsvm")
param = {'task':'binary', 'lr':0.0001, 'lambda':0.01, 'k':8, 'epoch':150}
xfm.fit(param, 'model.out')
xfm.setTXTModel("model.txt")

xfm.setSigmoid()
xfm.setTest("test.libsvm")
xfm.predict('model.out', "output.txt")

train = pd.read_csv('D:/1007studying/2021/DATA/test.csv', dtype={'id': 'U'})
click = pd.read_csv('output.txt', names=['click'])
out = pd.DataFrame({'id': train['id'], 'click': click['click']})
out.to_csv('D:/1007studying/2021/DATA/results.csv', index=False, sep=',')

