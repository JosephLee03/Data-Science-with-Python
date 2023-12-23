import pandas as pd
import numpy as np

def gini(data):
    """
    计算基尼指数
    :param data: 数据集，dataframe格式，最后一列为标签列
    :return: 基尼系数
    """

    data_label = data.iloc[:, -1]
    label_num = data_label.value_counts()  # 统计各类别的数量
    res = 0
    for k in label_num.keys():
        p_k = label_num[k] / len(data_label)  # 计算各类别出现的概率
        res += p_k ** 2

    return 1 - res

# 计算每个特征取值的基尼指数，
def gini_index(data,a):
    feature_class = data[a].value_counts()
    res = []
    for feature in feature_class.keys():
        weight = feature_class[feature]/len(data)
        gini_value = gini(data.loc[data[a] == feature])
        res.append([feature, weight * gini_value])
    res = sorted(res, key = lambda x: x[-1])
    return res[0]


#获取标签最多的那一类
def get_most_label(data):
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

#挑选最优特征，即基尼指数最小的特征
def get_best_feature(data):
    features = data.columns[:-1]
    res = {}
    for a in features:
        temp = gini_index(data, a) #temp是列表，【feature_value, gini】
        res[a] = temp
    res = sorted(res.items(),key=lambda x:x[1][1])
    return res[0][0], res[0][1][0]

def drop_exist_feature(data, best_feature, value, type):
    attr = pd.unique(data[best_feature]) #表示特征所有取值的数组
    if type == 1: #使用特征==value的值进行划分
        new_data = [[value], data.loc[data[best_feature] == value]]
    else:
        new_data = [attr, data.loc[data[best_feature] != value]]
    new_data[1] = new_data[1].drop([best_feature], axis=1) #删除该特征
    return new_data

#创建决策树
def create_tree(data):
    data_label = data.iloc[:,-1]
    if len(data_label.value_counts()) == 1: #只有一类
        return data_label.values[0]
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:,:-1].columns): #所有数据的特征值一样，选样本最多的类作为分类结果
        return get_most_label(data)
    best_feature, best_feature_value = get_best_feature(data) #根据信息增益得到的最优划分特征
    Tree = {best_feature:{}} #用字典形式存储决策树

    Tree[best_feature][best_feature_value] = create_tree(drop_exist_feature(data, best_feature, best_feature_value, 1)[1])
    Tree[best_feature]['Others'] = create_tree(drop_exist_feature(data, best_feature, best_feature_value, 2)[1])
    return Tree

def predict(Tree , test_data):
    first_feature = list(Tree.keys())[0] #第一个特征
    second_dict = Tree[first_feature] #第一个特征后面的字典
    input_first = test_data.get(first_feature) #预测输入的第一个特征值是多少
    input_value = second_dict[input_first] if input_first == list(second_dict.keys())[0] else second_dict['Others'] #预测输入对应的字典
    if isinstance(input_value , dict): #判断分支还是不是字典
        class_label = predict(input_value, test_data)
    else:
        class_label = input_value
    return class_label

if __name__  == '__main__':
    print("======= DT_gini.py主程序测试 =======")