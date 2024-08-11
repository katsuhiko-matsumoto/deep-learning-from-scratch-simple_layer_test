import sys, os
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.params['cnt'] = 0

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        self.params['cnt'] = self.params['cnt'] + 1
        #print('--------')
        #print('x:',x.shape)
        #print('W1:',W1.shape)
        a1 = np.dot(x, W1) + b1
        #print('W1-out:',a1.shape)
        z1 = sigmoid(a1)
        #print('W1-out(sigmoid):',z1.shape)
        #print('W2:',W2.shape)
        a2 = np.dot(z1, W2) + b2
        #print('W2-out:',a2.shape)
        #print('cnt:',self.params['cnt'])
        y = softmax(a2)

        return y

    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:入力データ, t:教師データ
    #順伝播
    def numerical_gradient(self, x, t):
        #print('#call numerical_gradient')
        loss_W = lambda W: self.loss(x, t)

        #print('#call recursive numerical_gradient')
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        #print('#call recursive W1 end:')
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        #print('#call recursive b1 end:')
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        #print('#call recursive W2 end:')
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        #print('#call recursive b2 end:')
        return grads

    #逆伝播
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
