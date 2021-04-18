import csv

import numpy as np

from scipy.stats import pearsonr

def expectation(line):
    buffer = 0.0
    for i in range(len(line)):
        buffer += line[i]
    buffer /= len(line)
    return buffer

def Cov(line_x, line_y):
    #Ex = expectation(line_x)
    Ex = line_x.mean()
    Ey = line_y.mean()
    #Ey = expectation(line_y)
    buffer = 0.0
    length = len(line_x)
    for i in range(length):
        buffer += (line_x[i]-Ex)*(line_y[i]-Ey)
    buffer /= length
    return buffer

def standard_Devia(line):
    buffer = 0.0
    E = line.mean()
    for i in range(len(line)):
        buffer += (line[i]-E)**2
    buffer /= len(line)
    buffer = buffer ** 0.5
    return buffer

def pearson_sum(matrix, index):
    number = len(matrix)
    result = []
    for i in range(len(matrix)):
        #if (i%100)==0:
            #print('find friends:%d/%d'%(i, len(matrix)))
        #buffer = Cov(np.array(matrix[i]), np.array(matrix[index]))/(np.array(matrix[i]).std()
        #*np.array(matrix[index]).std())
        #print('matrix[i]', matrix[i])
        #print('matrix[index]', matrix[index])
        buffer = pearsonr(matrix[i], matrix[index])[0]
        if i!=index:
            result.append(buffer)
        else:
            result.append(1)
    return result

def load_data(file_csv, header):
    data = []
    with open(file_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        if header!=None:
            headers = next(csv_reader)
        for row in csv_reader:
            buffer = row
            for i in range(len(buffer)):
                buffer[i] = eval(buffer[i])
            data.append(buffer)
    if header==None:
        return data
    else:
        return (headers, data)

def create_matrix(train_data, user_num, item_num, movie_alias):
    '''
    :param train_data:
    :param user_num:
    :param item_num:
    :param movie_alias:
    :return:
    注意user和item的编号都是从1开始的因此在生成矩阵的时候都要进行-1操作
    '''
    matrix = np.zeros([user_num, item_num])# 生成一个user*item的矩阵
    for i in range(len(train_data)):
        user_number = train_data[i][0]-1
        item_number = movie_alias[train_data[i][1]] - 1
        matrix[user_number][item_number] = train_data[i][2]
    return matrix

import heapq

def get_max_k_index(alist, k):
    new_alist = sorted(alist.copy(), reverse=True)
    k_data = new_alist[:k]
    max_index = []
    for x in k_data:
        max_index.append(alist.index(x))
    return max_index, k_data

#首先根据用户和matrix找到最接近的k个用户，主要不包含用户本身
def find_friends(user, matrix, k):
    user_index = user-1
    pearson_score = pearson_sum(matrix, user_index)
    friends_k, friends_k_score = get_max_k_index(pearson_score, k+1)[0][1:], get_max_k_index(pearson_score, k+1)[1][1:]
    return friends_k, friends_k_score

#根据这k个用户的评分情况对当前所有未评分的电影进行评分预测

def score_sum(friends_list, matrix, pearson_score, user_index):
    predict_list = []
    length = len(matrix[0])
    for i in range(length):
        buffer = 0.0
        score_sum = 0.0
        if matrix[user_index][i]==0:
            for j in range(len(friends_list)):
                buffer += pearson_score[j]*matrix[friends_list[j]][i]
                score_sum += pearson_score[j]
            buffer /= score_sum
            predict_list.append(buffer)
        else:
            predict_list.append(-1)
    return predict_list

def predict(user, matrix, k, n):
    friends_list, pearson_score = find_friends(user, matrix, k)
    user_index = user-1
    print('friends_list', friends_list)
    print('pearson_score', pearson_score)
    length = len(matrix[0])
    predict_list = score_sum(friends_list, matrix, pearson_score, user_index)
    result, _ = get_max_k_index(predict_list, n)
    return result

def find_friends_eval(user, matrix, movie, k):
    user_index = user-1
    movie_index = movie-1
    new_matrix = []
    user_dict = {}#映射的是下标
    indexs = 0
    for i in range(len(matrix)):
        if matrix[i][movie_index]!=0:
            if i!=user_index:
                new_matrix.append(matrix[i])
                user_dict[indexs]=i
                indexs+=1
    new_matrix.append(matrix[user_index])
    user_dict[indexs]=user_index
    indexs += 1
    print('new matrix_user_num', indexs)
    friends, friends_score = find_friends(indexs, new_matrix, k)
    for i in range(len(friends)):
        friends[i] = user_dict[friends[i]]
    return friends, friends_score

def eval_single(user, movie, matrix, k):
    friends_list, pearson_score = find_friends_eval(user, matrix, movie, k)
    eval_result = 0.0
    movie_index = movie-1

    score_sum = 0.0
    for i in range(len(friends_list)):
        eval_result += matrix[friends_list[i]][movie_index]*pearson_score[i]
        score_sum += pearson_score[i]
    eval_result /= score_sum
    return eval_result


def sheru(number):
    limit1 = int(number)
    limit2 = limit1+0.5
    if number-limit1>=0.25:
        if number-limit2>=0.25:
            return limit1+1
        else:
            return limit2
    else:
        return limit1

def eval_data(test_data, matrix, alias_movie, alias_movie_reverse, k_times):
    test_users = []
    test_movies = []
    real_result = []

    for i in range(len(test_data)):
        test_users.append(test_data[i][0])
        test_movies.append(alias_movie[test_data[i][1]])
        real_result.append(test_data[i][2]/5.0)

    pre_result = []
    for i in range(len(test_data)):
        pre_result.append(eval_single(test_users[i], test_movies[i], matrix, k_times)*5)

    loss = 0.0
    old_pre_result = pre_result.copy()

    for i in range(len(pre_result)):
        pre_result[i] = sheru(pre_result[i])

    for i in range(len(test_data)):
        if (i%10)==0:
            print('eval:i:%d'%i)
        loss += (pre_result[i]-(real_result[i])*5)**2
    #loss *=25
    for i in range(len(real_result)):
        real_result[i]*=5.0
    print(old_pre_result)
    print(pre_result)
    print(real_result)
    return loss




train_headers, train_data = load_data('../datasets/train_set.csv', 1)
print(train_headers)
#print(len(train_data))#99904
#print(train_data[0])
test_headers, test_data = load_data('../datasets/test_set.csv', 1)
#print(test_headers)
#print(len(test_data))
#print(test_data[0])

user_dict = {}
user_list = []
movie_dict = {}
movie_list = []


alias_user = {}
alias_movie = {}
alias_movie_reverse = {}

for i in range(len(train_data)):
    user_dict[train_data[i][0]] = user_dict.get(train_data[i][0], 0)+1
    movie_dict[train_data[i][1]] = movie_dict.get(train_data[i][1], 0)+1

print(len(user_dict))#671

for x in user_dict:
    user_list.append(x)

user_list = sorted(user_list, reverse=True)
print(user_list)#1~671


print(len(movie_dict))#9066
for x in movie_dict:
    movie_list.append(x)

#movie_list = sorted(movie_list, reverse=True)
#print(movie_list)#电影需要重新映射#[163949, 162672, 162542, 162376, 161944, 161918, 161830, 161594,
movie_list = sorted(movie_list)#从低到高映射
print(movie_list)

for i in range(len(movie_list)):
    alias_movie[movie_list[i]]=i+1

for x in alias_movie:
    alias_movie_reverse[alias_movie[x]] = x

#print(alias_movie)#将1～163949映射到1～9066

#user:1~671
#movie:1~9066
#matrix:(671X9066)



matrix = np.zeros([587, 9066])

train_score_max = 0


for i in range(len(train_data)):
    if train_data[i][2]>train_score_max:
        train_score_max=train_data[i][2]

print('max_score:', train_score_max)#5.0

for i in range(len(train_data)):
    train_data[i][2] = train_data[i][2]/5.0#归一化

matrix = create_matrix(train_data, len(user_dict), len(movie_dict), alias_movie)
print(matrix)

#item_list_pre = predict(452, matrix)
import time

begin = time.process_time()
loss_list = []
x_label = []
y_label = []
for k_times in range(3, 50):
    loss = eval_data(test_data, matrix, alias_movie, alias_movie_reverse, k_times)
    loss_list.append([k_times, loss])
    x_label.append(k_times)
    y_label.append(loss)
print(loss_list)
end = time.process_time()
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt

plt.xlabel('K')
plt.ylabel('loss')
plt.title('k-loss曲线')
plt.plot(x_label, y_label)
plt.show()
print('times{}ms'.format(end-begin))


'''
item_list_pre = predict(452, matrix, 5, 3)
#print(item_list_pre)
true_item_list_pre = []

for x in item_list_pre:
    true_item_list_pre.append(alias_movie_reverse[x+1])

print(true_item_list_pre)#[50, 1729, 21]
'''


