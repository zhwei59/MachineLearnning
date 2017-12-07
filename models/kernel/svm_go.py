import numpy as np
class perceptron:
    def __init__(self):
        self._w=self._b=None
    def fit(self,x,y,lr=0.01,epoch=10000):
        x,y=np.asarray(x,np.float32),np.asarray(y,np.float32)
        self._w=np.zeros(x.shape[1])
        self._b=0
        for i in range(epoch):
            err=-y*self.predict(x,True)
            idx=np.argmax(err)
            print  str(i) + "___" + str(self._w) + "___err___" + str(err[idx])
            if err[idx]<0:
                break
            delta=lr*y[idx]
            self._w+=delta*x[idx]
            self._b+=delta

    def predict(self,x,raw=False):
        x=np.asarray(x,np.float32)
        pred=x.dot(self._w)+self._b
        if raw:
            return pred
        return np.sign(pred).astype(np.float32)
class LinearSVM:
    def __init__(self):
        self._w=self._b=None
    def predict(self,x,raw=False):
        x=np.asarray(x,np.float32)
        pred=x.dot(self._w)
        if raw:
            return pred
        return np.sign(pred).astype(np.float32)
    def fit(self,x,y,c=1,lr=0.01,epoch=10000):
        x,y=np.asarray(x,np.float32),np.asarray(y,np.float32)
        self._w=np.zeros(x.shape[1])
        self._b=0
        for i in range(epoch):
            self._w*=1-lr

            err=1-y*self.predict(x,True)
            idx=np.argmax(err)

            if err[idx]<=0:
                print  str(i) + "___" + str(self._w) + "___err___" + str(err[idx])
                continue
            delta=lr*c*y[idx]
            self._w+=delta*x[idx]
            self._b+=delta