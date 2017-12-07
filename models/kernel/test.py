# -*- coding: UTF-8 -*-
from utils import gen_two_clusters
from svm_go import *

def show(x,y,model):
    print model._w, model._b
    print("ss:{:8.6} %".format((model.predict(x) == y).mean() * 100))
    from utils import visualize2d

    visualize2d(model, x, y)
    visualize2d(model, x, y, True)
x, y = gen_two_clusters(center=1,scale=5,dis=2)
x-=x.mean(axis=0)
x/=x.std(axis=0)
svm =perceptron ()
svm.fit(x, y)

show(x,y,svm)

svm2=LinearSVM()
svm2.fit(x,y)


show(x,y,svm2)
