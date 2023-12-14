import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class myANNClassifier:
    def __init__(self, layers = [3, 4, 2], lr = 0.01, activation = "sigmoid"):
        """
        params:
            layers: 神经网络层数、神经元个数。
            lr: 学习率。
            activation: 激活函数。
        """

        self.layers = layers
        self.num_layers = len(layers)
        self.weight_num = sum(map(lambda x, y: x * y, layers, layers[1:]))
        self.weights = [np.random.randn(x, y) * 0.5 for x, y in zip(layers[:-1], layers[1:])]   # list: weights[0]是第一层到第二层的权重, 以此类推
        self.bias = [np.random.randn(y, 1) * 0.5 for y in layers[1:]]  # list: bias[0]是第二层的偏置, 以此类推
        self.lr = lr
        self.activation = activation


    def __activation_func(self, x):
        """
        激活函数
        params:
            x: 输入数据，shape = (n, layers[0])。
        return:
            o: 输出数据，shape = (n, layers[-1])。
        """

        try:
            if self.activation == "sigmoid":
                return 1 / (1 + np.exp(-x))
            elif self.activation == "tanh":
                return np.tanh(x)
            elif self.activation == "relu":
                return np.maximum(0, x)
            else:
                raise ValueError("未找到您指定的激活函数")
        except ValueError as e:
            print(e)
            exit(1)


    # 每个样本的前向传播过程
    def forward(self, x):
        """
        前向传播过程
        params:
            x: 输入数据，shape = (1, layers[0])。
        return:
            output: 输出数据，shape = (n, layers[-1])。
        """

        x_c = x.copy().flatten()
        # 第一层初始化
        # 每个单元的输入，初始先全未0
        I = [np.zeros(i) for i in self.layers]
        # 每个单元的输出，初始先全未0
        O = [np.zeros(i) for i in self.layers]

        I[0] = x_c.reshape(-1,1)
        O[0] = x_c.reshape(-1,1)
       
        # 隐藏层+输出层的输入+矩阵乘法优化
        for j in range(1, self.num_layers):
            I[j] = np.dot(self.weights[j - 1].T, O[j - 1]).reshape(-1,1) + self.bias[j - 1]
            O[j] = self.__activation_func(I[j])

        return I, O


    def cal_err(self, y, O):
        """
        计算隐藏层和输出层每个神经元的误差
        params:
            y: 输出数据，shape = (1, layers[-1])。
            O: 样本输入后网络后，每个单元的输出。
        return:
            err: 隐藏层和输出层每个神经元的误差。
        """

        y_c = y.copy()

        # 初始化误差，全部为0.0。输入层没有err，但是为了方便计算，也初始化为0
        err = [np.zeros(i) for i in self.layers]
        err[0] = err[0].reshape(1, -1)
        
        for layer in range(self.num_layers - 1, 0, -1):
            if layer == self.num_layers - 1:
                # 输出层的误差
                err[layer] = (y_c - O[layer]) * O[layer] * (1 - O[layer])

            else:
                # 隐藏层的误差
                err[layer] = np.dot(err[layer + 1], self.weights[layer].T) * O[layer].T * (1 - O[layer].T)

        return err

    def backward(self, x, y):
        """
        反向传播过程，更新权重和偏置
        params:
            x: 输入数据，shape = (1, layers[0])。
            y: 输出数据，shape = (1, layers[-1])。"""
        
        
        I, O = self.forward(x)
        err = self.cal_err(y, O)
        for layer in range(self.num_layers-1):
            # 更新偏置
            self.bias[layer] += self.lr * err[layer+1].T    # 注意这里err[layer+1]只是因为bias初始化的时候没有考虑输入层
            # 更新权重
            self.weights[layer] += self.lr * np.dot(O[layer], err[layer+1])
        
        return I, O
           

    def train(self, x, y, epochs=10):
        """
        训练模型并计算损失
        params:
            x: 输入数据，shape = (n, layers[0])。
            y: 输出数据，shape = (n, layers[-1])。
            epochs: 迭代次数。
        """

        x_c = x.copy()
        y_c = y.copy()
        data_size = x_c.shape[0]
        losses = []  # 存储每次迭代的损失值

        for epoch in range(epochs):
        
            epoch_loss = 0.0  # 每个epoch损失值

            for i in range(data_size):
                _, outputs = self.backward(x_c[i:i+1], y_c[i:i+1])
                epoch_loss += np.sum((outputs[-1] - y_c[i:i+1][0]) ** 2) / 2

            epoch_loss /= data_size
            losses.append(epoch_loss)

            if epoch % 5 == 0:
                print("-----Epoch: {} Loss-----: {}".format(epoch, epoch_loss))


    def predict(self, x):
        """
        预测
        params:
            x: 输入数据，shape = (n, layers[0])。
        return:
            outputs: 预测值，shape = (n, layers[-1])。
        """

        x_c = x.copy()
        data_size = x_c.shape[0]
        outputs = []
        for i in range(data_size):
            print("predicting..., sample: {}".format(i+1), end="\r")
            _, output = self.forward(x_c[i:i+1])
            outputs.append(output[-1].item())

        return outputs


if __name__ == "__main__":
    print("====== myANNClassifier.py Test ======")
    x = np.random.randn(70, 5)
    y = np.square(np.mean(x, axis=1)).reshape(-1, 1)


    ann = myANNClassifier(layers = [x.shape[1], 3, 1], lr = 0.1, activation="sigmoid")
    ann.train(x, y, epochs=1000)

    z = np.random.randn(30, 5)

    label = np.square(np.mean(z, axis=1)).reshape(-1, 1)
    labels = list(label.reshape(1,-1).flatten())
    outputs = ann.predict(z)

    plt.figure(figsize=(10, 7))
    plt.plot(outputs, label="predict")
    plt.plot(labels, label="label")
    plt.legend()
    plt.show()
