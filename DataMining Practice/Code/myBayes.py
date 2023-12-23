import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = None     # 类别先验概率
        self.feature_probabilities = None   # 特征条件概率

    def fit(self, X, y):

        print("======= Naive Bayes Classifier fit start ======")

        n_samples, n_features = X.shape
        self.class_probabilities = {}  # 存储类别的先验概率
        self.feature_probabilities = {}  # 存储特征的条件概率

        classes = np.unique(y)  # 获取类别的唯一值
        for c in classes:
            X_c = X[y == c]  # 获取属于当前类别的样本
            self.class_probabilities[c] = len(X_c) / len(X)  # 计算类别的先验概率

            # 计算每个特征在当前类别下每个取值的条件概率
            self.feature_probabilities[c] = {}


            for feature in range(n_features):
                # print(X_c[:, feature])
                feature_counts = X_c.iloc[:, feature].value_counts().to_dict()  # 统计每个特征值出现的次数
                total_count = len(X_c)
                for value, count in feature_counts.items():
                    if feature not in self.feature_probabilities[c]:
                        self.feature_probabilities[c][feature] = {}
                    self.feature_probabilities[c][feature][value] = count / total_count  # 计算条件概率
        
        print("======= Naive Bayes Classifier fit finished ======")
        print("====== 类别先验概率 ======")
        print(self.class_probabilities)
        print("====== 特征条件概率 ======")
        print(self.feature_probabilities)


    def calculate_probability(self, x, feature_probabilities):
        # 计算样本 x 在给定特征条件概率下的概率
        prob = 1
        for feature, value in x.items():
            if value in feature_probabilities.get(feature, {}):
                prob *= feature_probabilities[feature][value]
        return prob

    def predict(self, X):
        
        print("=== Naive Bayes Classifier predict start ===")
        # print(X)
        predictions = []
        for sample in range(len(X)):
            
            posterior_probs = {}  # 存储后验概率
            for c in self.class_probabilities:
                prior = self.class_probabilities[c]  # 获取类别的先验概率
                likelihood = self.calculate_probability(X.iloc[sample, :], self.feature_probabilities[c])  # 计算似然度
                posterior_probs[c] = prior * likelihood  # 计算后验概率
            predictions.append(max(posterior_probs, key=posterior_probs.get))  # 选取后验概率最大的类别作为预测结果
        return predictions


def holdout(data, test_size=1/3):
    train, test = train_test_split(data, test_size=test_size)
    return train, test



if __name__ == "__main__":


    print("===== Naive Bayes Classifier =====")

    sampleData = pd.read_excel('./Data/附件1-Sampledata-discrete.xls').set_index('Id')
    data = sampleData.iloc[:-1, :]
    new_data = sampleData.iloc[-1, :-1]
    # 数值化处理
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data_encoded = data.apply(le.fit_transform)
    new_data_encoded = le.fit_transform(new_data)
    train, test = holdout(data_encoded)
    print(train.shape, test.shape)


    model = NaiveBayesClassifier()


    model.fit(train.iloc[:, :-1], train.iloc[:, -1])

    pre = model.predict(test.iloc[:, :-1])
    print('模型预测结果', pre)

