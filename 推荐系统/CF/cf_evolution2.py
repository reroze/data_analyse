import csv
import time
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

def mini_matrix_make(matrix):
    number = len(matrix)
    length = len(matrix)
    new_matrix = np.zeros([number, length])
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j]<=2.5:
                new_matrix[i][j] = 0
            else:
                new_matrix[i][j] = 1
    return new_matrix

import scipy.spatial.distance as dist

def jaccard_sum_single(a, b):
    ds = dist.pdist([a, b], 'jaccard')
    return ds[0]

def jaccard_sum(mini_matrix, user_index):
    jaccard_score_all = []
    for i in range(len(mini_matrix)):
        if i!=user_index:
            jaccard_score_buffer = jaccard_sum_single(mini_matrix[user_index], mini_matrix[i])
            jaccard_score_all.append([i, jaccard_score_buffer])
    sorted_jaccard_score_all = sorted(jaccard_score_all, key=lambda x:x[1], reverse=True)
    return sorted_jaccard_score_all

def jaccard_hash_sum_single(a, b):
    length = len(a)
    buffer = np.sum(a==b)
    return buffer/length

def jaccard_hash_sum(mini_hash_matrix, user_index):
    jaccard_score_all = []
    for i in range(len(mini_hash_matrix)):
        if i!=user_index:
            jaccard_score_buffer = jaccard_hash_sum_single(mini_hash_matrix[user_index], mini_hash_matrix[i])
            jaccard_score_all.append([i, jaccard_score_buffer])
    sorted_jaccard_score_all = sorted(jaccard_score_all, key=lambda x:x[1], reverse=True)
    return sorted_jaccard_score_all

def find_friends_mini(user, mini_matrix, k):
    user_index = user-1
    sorted_jaccard_score = jaccard_sum(mini_matrix, user_index)
    friends_k = []
    friends_k_score = []
    print('len(sorted_jaccard)', len(sorted_jaccard_score))
    number = min(k, len(mini_matrix)-1)
    for i in range(number):
        friends_k.append(sorted_jaccard_score[i][0])
        friends_k_score.append(sorted_jaccard_score[i][1])
    return friends_k, friends_k_score

def find_friends_mini_hash(user, mini_hash_matrix, k):
    user_index = user-1
    sorted_jaccard_score = jaccard_hash_sum(mini_hash_matrix, user_index)
    friends_k = []
    friends_k_score = []
    print('len(sorted_jaccard)', len(sorted_jaccard_score))
    number = min(k, len(mini_hash_matrix)-1)
    for i in range(number):
        friends_k.append(sorted_jaccard_score[i][0])
        friends_k_score.append(sorted_jaccard_score[i][1])
    return friends_k, friends_k_score
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

def score_sum(friends_list, matrix, pearson_score):
    predict_list = []
    length = len(matrix[0])
    for i in range(length):
        buffer = 0.0
        score_sum = 0.0
        for j in range(len(friends_list)):
            buffer += pearson_score[j]*matrix[friends_list[j]][i]
            score_sum += pearson_score[j]
        buffer /= score_sum
        predict_list.append(buffer)
    return predict_list

def score_sum_mini(friends_list, matrix, jaccard_score, user_index):
    predict_list = []
    length = len(matrix[0])#电影数目

    for i in range(length):
        buffer = 0.0
        score_sum = 0.0
        if matrix[user_index][i]==0:
            for j in range(len(friends_list)):
                buffer += jaccard_score[j]*matrix[friends_list[j]][i]
                score_sum += jaccard_score[j]
            if(score_sum==0):
                buffer=0.0
            else:
                buffer /= score_sum
            predict_list.append(buffer)
        else:
            predict_list.append(-1)
    return predict_list

def predict_mini(user, matrix, mini_matrix, k, n):
    #friends_list, pearson_score = find_friends(user, matrix, 5)
    friends_list, jaccard_score = find_friends_mini(user, mini_matrix, k)
    print('friends_list', friends_list)
    print('jaccard_score', jaccard_score)
    length = len(matrix[0])
    predict_list = score_sum_mini(friends_list, matrix, jaccard_score, user-1)
    result, _ = get_max_k_index(predict_list, n)
    return result

def predict_mini_hash(user, matrix, mini_hash_matrix, k, n):
    #friends_list, pearson_score = find_friends(user, matrix, 5)
    friends_list, jaccard_score = find_friends_mini_hash(user, mini_hash_matrix, k)
    print('friends_list', friends_list)
    print('jaccard_score', jaccard_score)
    length = len(matrix[0])
    predict_list = score_sum_mini(friends_list, matrix, jaccard_score, user-1)
    result, _ = get_max_k_index(predict_list, n)
    return result

def predict(user, matrix):
    friends_list, pearson_score = find_friends(user, matrix, 5)
    print('friends_list', friends_list)
    print('pearson_score', pearson_score)
    length = len(matrix[0])
    predict_list = score_sum(friends_list, matrix, pearson_score)
    result, _ = get_max_k_index(predict_list, 3)
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

def find_friends_eval_mini(user, matrix, mini_matrix, movie, k):
    user_index = user-1
    movie_index = movie-1
    new_matrix = []
    new_matrix_mini = []
    user_dict = {}#映射的是下标
    indexs = 0
    for i in range(len(matrix)):
        if matrix[i][movie_index]!=0:
            if i!=user_index:
                new_matrix.append(matrix[i])
                new_matrix_mini.append(mini_matrix[i])
                user_dict[indexs]=i
                indexs+=1
    new_matrix.append(matrix[user_index])
    new_matrix_mini.append(mini_matrix[user_index])
    user_dict[indexs]=user_index
    indexs += 1
    if(indexs<28):
        print('user_index', user_index)
        print('movie_index', movie_index)
    print('new matrix_user_num', indexs)
    friends, friends_score = find_friends_mini(indexs, new_matrix_mini, k)
    for i in range(len(friends)):
        friends[i] = user_dict[friends[i]]
    return friends, friends_score

def find_friends_eval_mini_hash(user, matrix, mini_hash_matrix, movie, k):
    user_index = user-1
    movie_index = movie-1
    new_matrix = []
    new_matrix_mini_hash = []
    user_dict = {}#映射的是下标
    indexs = 0
    for i in range(len(matrix)):
        if matrix[i][movie_index]!=0:
            if i!=user_index:
                new_matrix.append(matrix[i])
                new_matrix_mini_hash.append(mini_hash_matrix[i])
                user_dict[indexs]=i
                indexs+=1
    new_matrix.append(matrix[user_index])
    new_matrix_mini_hash.append(mini_hash_matrix[user_index])
    user_dict[indexs]=user_index
    indexs += 1
    if(indexs<28):
        print('user_index', user_index)
        print('movie_index', movie_index)
    print('new matrix_user_num', indexs)
    friends, friends_score = find_friends_mini_hash(indexs, new_matrix_mini_hash, k)
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



def eval_single_mini(user, movie, matrix, mini_matrix, k):
    friends_list, jaccard_score = find_friends_eval_mini(user, matrix, mini_matrix, movie, k)
    eval_result = 0.0
    movie_index = movie-1

    score_sum = 0.0
    for i in range(len(friends_list)):
        eval_result += matrix[friends_list[i]][movie_index]*jaccard_score[i]
        score_sum += jaccard_score[i]
    eval_result /= score_sum
    return eval_result

def eval_single_mini_hash(user, movie, matrix, mini_hash_matrix, k):
    friends_list, jaccard_score = find_friends_eval_mini_hash(user, matrix, mini_hash_matrix, movie, k)
    eval_result = 0.0
    movie_index = movie-1

    score_sum = 0.0#此处要设置成对应1/hash函数数目
    for i in range(len(friends_list)):
        eval_result += matrix[friends_list[i]][movie_index]*jaccard_score[i]
        score_sum += jaccard_score[i]
    if score_sum==0.0:
        eval_result=2.5
    else:
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

def eval_data(test_data, matrix, alias_movie, alias_movie_reverse):
    test_users = []
    test_movies = []
    real_result = []

    for i in range(len(test_data)):
        test_users.append(test_data[i][0])
        test_movies.append(alias_movie[test_data[i][1]])
        real_result.append(test_data[i][2]/5.0)

    pre_result = []
    for i in range(len(test_data)):
        pre_result.append(eval_single(test_users[i], test_movies[i], matrix, 28)*5)

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

def eval_data_mini(test_data, matrix, alias_movie, mini_matrix, alias_movie_reverse):
    test_users = []
    test_movies = []
    real_result = []

    for i in range(len(test_data)):
        test_users.append(test_data[i][0])
        test_movies.append(alias_movie[test_data[i][1]])
        real_result.append(test_data[i][2])

    pre_result = []
    for i in range(len(test_data)):
        pre_result.append(eval_single_mini(test_users[i], test_movies[i], matrix, mini_matrix, 42))

    loss = 0.0
    old_pre_result = pre_result.copy()

    #for i in range(len(pre_result)):
        #pre_result[i] = sheru(pre_result[i])

    for i in range(len(test_data)):
        if (i%10)==0:
            print('eval:i:%d'%i)
        loss += (pre_result[i]-(real_result[i]))**2
    #loss *=25
    for i in range(len(real_result)):
        real_result[i]*=5.0
    print(old_pre_result)
    print(pre_result)
    print(real_result)
    return loss

def eval_data_mini_hash(test_data, matrix, alias_movie, mini_hash_matrix, alias_movie_reverse, k_times):
    test_users = []
    test_movies = []
    real_result = []

    for i in range(len(test_data)):
        test_users.append(test_data[i][0])
        test_movies.append(alias_movie[test_data[i][1]])
        real_result.append(test_data[i][2])

    print('real_result_old', real_result)

    pre_result = []
    for i in range(len(test_data)):
        pre_result.append(eval_single_mini_hash(test_users[i], test_movies[i], matrix, mini_hash_matrix, k_times))

    loss = 0.0
    old_pre_result = pre_result.copy()

    #for i in range(len(pre_result)):
        #pre_result[i] = sheru(pre_result[i])

    for i in range(len(test_data)):
        if (i%10)==0:
            print('eval:i:%d'%i)
        loss += (pre_result[i]-(real_result[i]))**2
    #loss *=25

    print(old_pre_result)
    print(pre_result)
    print(real_result)
    return loss

import random
import pickle

def random_hash(N):
    A = random.sample(range(1, 10*N), N)
    B = random.sample(range(1, 10*N), N)
    C = random.sample(range(5*N, 10*N), N)
    return A, B, C



class mini_hash():
    def __init__(self, N):
        self.N = N
        self.As, self.Bs, self.Cs = random_hash(N)
        self.Hashs = []
        for i in range(N):
            hash_buffer = lambda x:(self.As[i]*x+self.Bs[i])%self.Cs[i]
            self.Hashs.append(hash_buffer)

    def hash_single(self, i_index, j_index, hash_matrix):
        for i in range(self.N):
            hash_buffer = self.Hashs[i](j_index)
            if hash_buffer<hash_matrix[i_index][i]:
                hash_matrix[i_index][i]=hash_buffer

    def save_model(self, loss):
        save_file = 'good_hash_'+str(self.N)+'_'+str(loss)[:5]
        save_file_op = open(save_file, 'wb')
        pickle.dump([self.As, self.Bs, self.Cs], save_file_op)
        save_file_op.close()
        print("模型保存成功到"+save_file)

    def load_model(self, file):
        file_op = open(file, "rb")
        buffer = pickle.load(file_op)
        self.As = buffer[0]
        self.Bs = buffer[1]
        self.Cs = buffer[2]
        file_op.close()
        print("模型加载成功")

    def hash_matrix(self, matrix):
        '''

        :param matrix: (User,Matrix)
        :return:
        '''
        user_num = len(matrix)
        length = len(matrix[0])
        #hash_matrix = np.zeros(user_num, self.N)
        inf = 10*self.N+1
        hash_matrix = np.full([user_num, self.N], inf)
        for i in range(user_num):
            for j in range(length):
                if matrix[i][j]==1:
                    self.hash_single(i, j, hash_matrix)
        return hash_matrix






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

print('4_num:', movie_dict[4])

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
movie_list = sorted(movie_list)
print(movie_list)

for i in range(len(movie_list)):
    alias_movie[movie_list[i]]=i+1

for x in alias_movie:
    alias_movie_reverse[alias_movie[x]] = x

print('wrong movie_id', alias_movie_reverse[4])
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
'''
for i in range(len(train_data)):
    train_data[i][2] = train_data[i][2]/5.0#归一化
'''
for i in range(len(train_data)):
    train_data[i][2] = train_data[i][2]#归一化
#为计算mini矩阵先不归一化


matrix = create_matrix(train_data, len(user_dict), len(movie_dict), alias_movie)
#print(matrix[2][:250])
mini_matrix = mini_matrix_make(matrix)
#print(mini_matrix[2][:250])

'''
for i in range(20):
    a_mini_hash = mini_hash(50)#91.58807939097173#83.0958570155577
    #a_mini_hash.load_model('good_hash_50_88.44')
    a_hash_matrix = a_mini_hash.hash_matrix(mini_matrix)
    loss = eval_data_mini_hash(test_data, matrix, alias_movie, a_hash_matrix, alias_movie_reverse)
    print(loss)
    if loss<70:
        a_mini_hash.save_model(loss)
        break
'''

a_mini_hash = mini_hash(500)#91.58807939097173#83.0958570155577
a_mini_hash.load_model('good_hash_500_61.71')
a_hash_matrix = a_mini_hash.hash_matrix(mini_matrix)
'''
x_label = []
y_label = []

import time

begin = time.process_time()

for k_times in range(3, 50):
    loss = eval_data_mini_hash(test_data, matrix, alias_movie, a_hash_matrix, alias_movie_reverse, k_times)
    x_label.append(k_times)
    y_label.append(loss)

end = time.process_time()

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt

plt.xlabel('K')
plt.ylabel('loss')
plt.title('k-loss曲线')
plt.plot(x_label, y_label)
plt.show()
print(min(y_label))
print(y_label.index(min(y_label))+3)
print('times {}ms'.format(end-begin))
'''




'''
item_list_pre = predict(452, matrix)

true_item_list_pre = []

for x in item_list_pre:
    true_item_list_pre.append(alias_movie_reverse[x+1])

print(true_item_list_pre)

loss = eval_data(test_data, matrix, alias_movie, alias_movie_reverse)
print(loss)
'''

#item_list_pre = predict_mini(452, matrix, mini_matrix, 5, 3)
#print(item_list_pre)
#true_item_list_pre = []

#for x in item_list_pre:
    #true_item_list_pre.append(alias_movie_reverse[x+1])

#print(true_item_list_pre)#[3157, 73, 73]

item_list_pre = predict_mini_hash(452, matrix, a_hash_matrix, 20, 3)
#print(item_list_pre)
true_item_list_pre = []

for x in item_list_pre:
    true_item_list_pre.append(alias_movie_reverse[x+1])

print(true_item_list_pre)#[246, 50, 32]

