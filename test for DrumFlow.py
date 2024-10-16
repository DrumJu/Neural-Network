import DrumFLow
import numpy as np

from DrumFLow import get_prediction

data1=np.loadtxt("classify data.txt",delimiter=",",dtype=np.float64)

X_train,Y_train,m,n=DrumFLow.data_loader(data1)
Y_train=np.array(Y_train,dtype=int)


def trainmodel(iterations):
    w1, b1, w2, b2 = DrumFLow.initial_parameters(m, n, 2)
    for i in range(iterations):
       z1, z2, a1, a2 = DrumFLow.forward_propagation(X_train, w1, b1, w2, b2)
       dw1, db1, dw2, db2 = DrumFLow.back_propagation(X_train=X_train, Y_train=Y_train, m=m, z1=z1, a1=a1, a2=a2, w2=w2)
       w1, b1, w2, b2 = DrumFLow.update_parameters(dw1, db1, dw2, db2, w1, b1, w2, b2, 0.008)
       if i%1==0:
          prediction=get_prediction(a2)
          print(f"迭代次数：{i}")
          print(f"准确率：{DrumFLow.get_accuracy(get_prediction(a2),Y_train=Y_train)}")

    return w1,b1,w2,b2

trainmodel(50)



