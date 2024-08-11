import sys, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import time

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 繰り返しの回数を適宜設定する
#サンプル数：79523
train_size = x_train.shape[0]
batch_size = 100 #バッチサイズ
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

print('train_size:',train_size)
print('batch_size:',batch_size)
iter_per_epoch = max(train_size / batch_size, 1)

print('iter_per_epoch:',iter_per_epoch)
print('iter_num:',iters_num)

for i in range(iters_num):
    start = time.time()
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    #順伝播
    grad = network.numerical_gradient(x_batch, t_batch)
    #逆伝播
    #grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    epoch_time = time.time()-start
    print('Time for epoch {} is {} sec'.format(i + 1, epoch_time))
    print('Total Time: {}, Estimated time remaining {}'.format(epoch_time * iters_num,
        epoch_time * (iters_num-(i+1))))
# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
