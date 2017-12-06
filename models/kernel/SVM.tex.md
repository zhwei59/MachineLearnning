## SVM
## 感知机模型
感知机是个古老的模型，1975年就提出了，今天看来感知机分类的泛化能力不强，但是人们在基础模型上做了大量的修正，产生了一系列的增强模型，学习感知机模型有助于学习深度学习。

感知机的思想很简单，在m维空间找一个超平面把样本点二元分类完全分开。如果找不到这样的超平面意味着，样本不是线性可分的，这意味着感知机模型不适合你的数据。当然通过核技巧可以让数据在高维可分，神经网络通过激活函数和增加隐藏层来让数据可分。

感知机是个相当简单的模型，但它即发展成SVM(通过修改损失函数)，又可以发展成神经网络(通过简单的堆叠)，所以它拥有一定的地位。

## 感知机模型python简单实现
原始形式

```

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
```


对偶形式


```
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
```

## SVM分类器

- 线性可分，通过硬间隔最大化，学习线性分类器
- 线性近似可分，通过软间隔最大化，学习线性分类器
- 线性不可分，通过核函数和软间隔，学习非线性分类器。

不用数学是体会不到svm的精髓的，


## KTT条件
最终得到了SVM模型的策略

$\min\limits_{\gamma,w,b} \frac {1}{2} \lVert w \rVert$


$s.t \qquad y^{(i)}(w^Tx^{(i)}+b)\geq 1 ,i=1,2,3,...,N$ 


一般地，我们用拉格朗日来解决有约束的最优化问题。问题转换成对偶方式：


$L(w,\beta)=f(w)+\sum\limits_{i=1}^l\beta_ih(w_i)$

其中$\beta_i$叫做拉格朗日乘子。

$\frac {{\partial}L}  {{\partial}w_i}=0 ; \frac {{\partial}L} {{\partial}\beta_i}=0$

就可以求解出$w,\beta$

ktt条件的产生，是因为上述的转化并非无条件的，附加的限制条件就是KTT条件。


## 优化间隔分类器



