# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
data_path='/Users/zhwei/share/lixianglan/gitrepo/kaggle/digitRecognizer/dev.csv'
p_path='/Users/zhwei/share/lixianglan/gitrepo/kaggle/digitRecognizer/test.csv'
tdf=pd.read_csv(p_path)
t_data=tdf.loc[:, (tdf != 0).any(axis=0)]
v_cloums=t_data.columns
df=pd.read_csv(data_path)
ps=df[v_cloums]
ps.insert(0,'label',df['label'])
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=4, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=1800, learning_rate=0.97)
cols=len(ps.columns)
X=ps.ix[:,range(1,cols)]
Y=ps['label']
bdt.fit(X,Y)
result=bdt.predict(t_data)
pd.DataFrame(result, columns=['Label']).to_csv('/Users/zhwei/share/lixianglan/gitrepo/kaggle/digitRecognizer/result.csv', header=True, index=True)
