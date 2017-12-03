import sklearn.datasets as ds
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import k_means

iris=ds.load_iris()
X,y=iris.data,iris.target
data=X[:,[1,3]]
#plt.scatter(data[:,0],data[:,1])
#plt.show()


def kmeans(data,k=2):
    def _distance(p1,p2):
        t_sum=np.sum((p1-p2)**2)
        return np.sqrt(t_sum)
    def _rand_center(data,k):
        n=data.shape[1]
        centeroids=np.zeros((k,n))
        for i in range(k):
            dmin,dmax=np.min(data[:,i]),np.max(data[:,i])
            centeroids[:,i]=dmin+(dmax-dmin)*np.random.rand(k)
        return centeroids

    def _converged(centeroids1,centeroids2):
        set1=set([tuple(c) for c in centeroids1])
        set2=set([tuple(c) for c in centeroids2])
        return  set1==set2
    n=data.shape[0]
    centeroids=_rand_center(data,k)
    lable=np.zeros(n,dtype=np.int)
    assement=np.zeros(n)
    converged=False
    while not converged:
        old_centerioids=np.copy(centeroids)
        for i in range(n):
            min_dist=np.inf
            for j in range(k):
                cdis=_distance(data[i],centeroids[j])
                if cdis<min_dist:
                    min_dist=cdis
                    lable[i]=j
            assement[i]=_distance(data[i],centeroids[lable[i]])**2
        for j in range(k):
            centeroids[j]=np.mean(data[lable==j],axis=0)
        converged=_converged(centeroids,old_centerioids)
    return centeroids,lable,np.sum(assement)


best_assement = np.inf
best_centroids = None
best_label = None
centroids, label, assement = kmeans(data, 2)
for i in range(10):
    centroids, label, assement = kmeans(data,2)
    if assement < best_assement:
        best_assement = assement
        best_centroids = centroids
        best_label = label

data0 = data[best_label==0]
data1 = data[best_label==1]
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
ax1.scatter(data[:,0],data[:,1],c='c',s=30,marker='o')
ax2.scatter(data0[:,0],data0[:,1],c='r')
ax2.scatter(data1[:,0],data1[:,1],c='c')
ax2.scatter(centroids[:,0],centroids[:,1],c='b',s=120,marker='o')
plt.show()


