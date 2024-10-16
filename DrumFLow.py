import numpy as np
#输入数据类型必须第一列为标签，后面为参数
#朕只会线性层所以只写线性层
#为了拟合效果尽量用于简单线性分类
#Code——Creator：Drum JU
def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
#Softmax激活函数
def ReLu(x):
    return np.maximum(0,x)
#Relu激活函数
def deriv_ReLu(x):
    return x>0
#relu的导数(布尔型数组）

def data_loader(a_input):
    m,n = a_input.shape
    data=a_input.T
    Y_train=data[0]
    X_train=data[1:n]
    return X_train,Y_train,m,n
#x_train为n-1行，m列
#y_train为1行，m列
def initial_parameters(m,n,out_features):
        w1=np.random.rand(out_features,n-1)-0.5
        b1=np.random.rand(out_features,m)-0.5
        w2=np.random.rand(out_features,out_features)-0.5
        b2=np.random.rand(out_features,m)-0.5
        return w1,b1,w2,b2
    #初始化参数
def forward_propagation(x,w1,b1,w2,b2):
        z1=w1.dot(x) + b1
        a1=ReLu(z1)
        z2=w2.dot(a1) + b2
        a2=Softmax(z2)
        return z1,z2,a1,a2

#前向传播
def one_hot(y):
        one_hot_y=np.zeros((y.size,y.max()+1))
        one_hot_y[np.arange(y.size),y]=1
        one_hot_y=one_hot_y.T
        return one_hot_y
#独热 输出（m行，out——features列）的矩阵

def back_propagation(X_train,Y_train,m,z1,a1,a2,w2):
       onehot_y=one_hot(Y_train)
       dz2=a2-onehot_y
       dw2=1/m * dz2.dot(a1.T)
       db2=1/m * dz2
       dz1=w2.T.dot(dz2)*deriv_ReLu(z1)
       dw1=1/m * dz1.dot(X_train.T)
       db1=1/m *dz1
       return dw1,db1,dw2,db2
def update_parameters(dw1,db1,dw2,db2,w1,b1,w2,b2,learning_rate):
    w1=w1-learning_rate*dw1
    b1=b1-learning_rate*db1
    w2=w2-learning_rate*dw2
    b2=b2-learning_rate*db2
    return w1,b1,w2,b2
#更新参数
def get_prediction(a2):
    return np.argmax(a2,axis=0)
def get_accuracy(prediction,Y_train):
    return np.sum(prediction==Y_train)/Y_train.size
def make_predictions(x,w1,b1,w2,b2):
    _,_,_,a2= forward_propagation(x,w1,b1,w2,b2)
    prediction=get_prediction(a2)
    return prediction
def loss_function(prediction,Y_train):
    loss=(prediction-Y_train)**2
    return loss










