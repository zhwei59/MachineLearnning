import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def sign(v):
    if v>=0:
        return 1
    else:
        return -1

def train(train_datas):
    X_train=train_datas[:,0:-1]
    print X_train
    y=train_datas[:,-1]
    print y
    m,n=np.shape(X_train)
    w,b=np.zeros((1,n)),0
    while True:
        for i in range(m):
            res=y[i]*(np.dot(w,X_train[i])+b)
            if res<=0:
                for j in range(n):
                    w[0][j]+=y[i]*X_train[i][j]
                b+=y[i]
                break
        else :
            break
    return w,b

def dual_train(train_datas):
    X_train=train_datas[:,0:-1]
    y=train_datas[:,-1]

    m,n=np.shape(X_train)
    gamma=np.dot(X_train,X_train.T)

    alpha,b=[0 for __ in range(m)],0
    print alpha
    while True:
        for i in range(m):
            tmp=0
            for j in range(m):
                tmp+=alpha[j]*y[j]*gamma[i,j]
            res=y[i]*(tmp+b)
            if res<=0:
                alpha[i]+=1
                b+=y[i]
                print alpha,b
                break
        else:
            break
    w=np.zeros((1,n))
    for i in range(m):
        w+=alpha[i]*y[i]*X_train[i]

    print w,b
    return w,b
def plot_points(train_datas,w,b):
    plt.figure()
    x1 = np.linspace(0, 8, 100)
    x2=(x1*w[0][0]+b)/(-w[0][1])

    plt.plot(x1, x2, color='r', label='y1 data')
    datas_len=len(train_datas)
    for i in range(datas_len):
        if(train_datas[i][-1]==1):
            plt.scatter(train_datas[i][0],train_datas[i][1],s=50)
        else:
            plt.scatter(train_datas[i][0],train_datas[i][1],marker='x',s=50)
    plt.show()


if __name__=='__main__':
    #train_datas = np.array(([1, 3, 1], [2, 2, 1], [3, 8, 1], [2, 6, 1],[2, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]))
    train_datas=np.array(([3,3,1],[3,4,1],[1,1,-1]))
    w,b=dual_train(train_datas=train_datas)
    print w,b
    plot_points(train_datas,w,b)