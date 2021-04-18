import csv
import numpy as np

def load_data(file_csv, header):
    data = []
    with open(file_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        if header!=None:
            headers = next(csv_reader)
        for row in csv_reader:
            buffer = row
            buffer[2] = buffer[2].split('|')
            data.append(buffer)
    if header==None:
        return data
    else:
        return (headers, data)

def load_data_eval(file_csv, header):
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

def matrix_make(data, genres_hash):
    number = len(data)
    length = len(genres_hash)
    matrix = np.zeros([number, length])
    for i in range(number):
        genres_single = data[i][2]
        for x in genres_single:
            matrix[i][genres_hash[x]]=1
    return matrix

def tf_idf_make_single(matrix_line, matrix, genres_hash, genres_dict):
    tf_idf_line = np.zeros([len(matrix_line)])
    all_genres_num_single = sum(matrix_line)
    number = len(matrix)
    genres_hash_reverse = {}
    for x in genres_hash:
        genres_hash_reverse[genres_hash[x]] = x
    for i in range(len(matrix[0])):
        tf_idf_line[i] = (matrix_line[i]/all_genres_num_single)*np.log(number/genres_dict[genres_hash_reverse[i]])
    all_buffer = sum(tf_idf_line)
    tf_idf_line /= all_buffer
    return tf_idf_line

def tf_idf_make(matrix, genres_hash, genres_dict):
    number = len(matrix)
    length = len(matrix[0])
    tf_idf_matrix = np.zeros([number, length])
    for i in range(len(matrix)):
        tf_idf_matrix[i] = tf_idf_make_single(matrix[i], matrix, genres_hash, genres_dict)
    return tf_idf_matrix

def find_other_movie(user, movie, train_data):
    movie_rate_list = []
    for i in range(len(train_data)):
        if train_data[i][0]==user:
            if train_data[i][1]!=movie:
                movie_rate_list.append([train_data[i][1], train_data[i][2]])
    #print(movie_rate_list)
    return movie_rate_list

def cosine_similar(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom



def eval_single(test_data, train_data, tf_idf_matrix, movie_dict):
    real_result = []
    pre_result = []
    # movie_dict:真实->矩阵
    movie_dict_reverse = {}  # 矩阵->真实
    for x in movie_dict:
        movie_dict_reverse[movie_dict[x]] = x
    for i in range(1):
        movie_rate_list = find_other_movie(test_data[i][0], test_data[i][1], train_data)
        real_result.append(test_data[i][2])
        test_movie_juzhen_index = movie_dict[test_data[i][1]]
        pre_buffer = 0.0
        cosine_s_all = 0.0
        for k in range(len(movie_rate_list)):
            cosine_s = cosine_similar(tf_idf_matrix[movie_dict[movie_rate_list[k][0]]],
                                      tf_idf_matrix[test_movie_juzhen_index])
            if cosine_s < 0:
                cosine_s = 0
            cosine_s_all+=cosine_s
            # print(cosine_s)
            # print(movie_rate_list[k][1])
            pre_buffer += cosine_s * movie_rate_list[k][1]
            print(pre_buffer)
        pre_result.append(pre_buffer/cosine_s_all)

    loss = 0.0
    print(pre_result)
    for i in range(len(pre_result)):
        loss += (pre_result[i] - real_result[i]) ** 2
    print(loss)

import scipy.spatial.distance as dist

def jaccard_sum_single(a, b):
    ds = dist.pdist([a, b], 'jaccard')
    return ds[0]


def eval_single_mini(test_data, train_data, matrix, movie_dict):
    real_result = []
    pre_result = []
    # movie_dict:真实->矩阵
    movie_dict_reverse = {}  # 矩阵->真实
    for x in movie_dict:
        movie_dict_reverse[movie_dict[x]] = x
    for i in range(1):
        movie_rate_list = find_other_movie(test_data[i][0], test_data[i][1], train_data)
        real_result.append(test_data[i][2])
        test_movie_juzhen_index = movie_dict[test_data[i][1]]
        pre_buffer = 0.0
        #cosine_s_all = 0.0
        jaccard_all = 0.0
        for k in range(len(movie_rate_list)):
            #cosine_s = cosine_similar(tf_idf_matrix[movie_dict[movie_rate_list[k][0]]],
                                      #tf_idf_matrix[test_movie_juzhen_index])
            jaccard_s = jaccard_sum_single(matrix[movie_dict[movie_rate_list[k][0]]],
                                           matrix[test_movie_juzhen_index])
            jaccard_all+=jaccard_s
            # print(cosine_s)
            # print(movie_rate_list[k][1])
            pre_buffer += jaccard_s * movie_rate_list[k][1]
            print(pre_buffer)
        pre_result.append(pre_buffer/jaccard_all)

    loss = 0.0
    print(pre_result)
    for i in range(len(pre_result)):
        loss += (pre_result[i] - real_result[i]) ** 2
    print(loss)

def recom_score_sum(movie_rate_list_juzhen, tf_idf_matrix, index):
    cosine_s_all = 0.0
    buffer = 0.0
    for x in movie_rate_list_juzhen:
        cosine_s = cosine_similar(tf_idf_matrix[x[0]], tf_idf_matrix[index])
        if cosine_s<0:
            cosine_s=0
        cosine_s_all+=cosine_s
        buffer+=cosine_s*x[1]
    return buffer/cosine_s_all

def recom_score_sum_mini(movie_rate_list_juzhen, matrix, index):
    jaccard_s_all = 0.0
    buffer = 0.0
    for x in movie_rate_list_juzhen:
        jaccard_s = jaccard_sum_single(matrix[x[0]], matrix[index])
        jaccard_s_all+=jaccard_s
        buffer+=jaccard_s*x[1]
    return buffer/jaccard_s_all

def recom_score_sum_mini_hash(movie_rate_list_juzhen, hash_matrix, index):
    jaccard_s_all = 0.0
    buffer = 0.0
    for x in movie_rate_list_juzhen:
        jaccard_s = jaccard_hash_sum_single(hash_matrix[x[0]], hash_matrix[index])
        jaccard_s_all+=jaccard_s
        buffer+=jaccard_s*x[1]
    if jaccard_s_all==0:
        return 2.5
    else:
        return buffer/jaccard_s_all

def recommendation(userID, k, train_data, movie_dict, tf_idf_matrix):
    #movie_dict:从真实映射到矩阵
    movie_dict_reverse = {}
    for x in movie_dict:
        movie_dict_reverse[movie_dict[x]] = x
    movie_rate_list = find_other_movie(userID, -1, train_data)
    movie_rate_list_juzhen = []
    for x in movie_rate_list:
        movie_rate_list_juzhen.append([movie_dict[x[0]], x[1]])
    movie_watched = []
    movie_score = []
    for i in range(len(movie_rate_list)):
        movie_watched.append(movie_dict[movie_rate_list[i][0]])
    for i in range(len(tf_idf_matrix)):
        if (i%100)==0:
            print("times:%d"%i)
        if i not in movie_watched:
            buffer_score = recom_score_sum(movie_rate_list_juzhen, tf_idf_matrix, i)
            movie_score.append([i, buffer_score])
    new_movie_score = sorted(movie_score, key=lambda x:x[1], reverse=True)
    recommendation_result = []
    for i in range(k):
        recommendation_result.append(movie_dict_reverse[new_movie_score[i][0]])
    return recommendation_result#推荐的是没看过的

def recommendation_mini(userID, k, train_data, movie_dict, matrix):
    #movie_dict:从真实映射到矩阵
    movie_dict_reverse = {}
    for x in movie_dict:
        movie_dict_reverse[movie_dict[x]] = x
    movie_rate_list = find_other_movie(userID, -1, train_data)
    movie_rate_list_juzhen = []#找到用户观看的所有电影的id并将其转换为矩阵中的id并存起来
    for x in movie_rate_list:
        movie_rate_list_juzhen.append([movie_dict[x[0]], x[1]])
    movie_watched = []
    movie_score = []
    for i in range(len(movie_rate_list)):
        movie_watched.append(movie_dict[movie_rate_list[i][0]])
    for i in range(len(matrix)):
        if (i%100)==0:
            print("times:%d"%i)
        if i not in movie_watched:
            buffer_score = recom_score_sum_mini(movie_rate_list_juzhen, matrix, i)
            movie_score.append([i, buffer_score])
    new_movie_score = sorted(movie_score, key=lambda x:x[1], reverse=True)
    recommendation_result = []
    for i in range(k):
        recommendation_result.append(movie_dict_reverse[new_movie_score[i][0]])
    return recommendation_result#推荐的是没看过的

def recommendation_mini_hash(userID, k, train_data, movie_dict, hash_matrix):
    #movie_dict:从真实映射到矩阵
    movie_dict_reverse = {}
    for x in movie_dict:
        movie_dict_reverse[movie_dict[x]] = x
    movie_rate_list = find_other_movie(userID, -1, train_data)
    movie_rate_list_juzhen = []#找到用户观看的所有电影的id并将其转换为矩阵中的id并存起来
    for x in movie_rate_list:
        movie_rate_list_juzhen.append([movie_dict[x[0]], x[1]])
    movie_watched = []
    movie_score = []
    for i in range(len(movie_rate_list)):
        movie_watched.append(movie_dict[movie_rate_list[i][0]])
    for i in range(len(matrix)):
        if (i%100)==0:
            print("times:%d"%i)
        if i not in movie_watched:
            buffer_score = recom_score_sum_mini_hash(movie_rate_list_juzhen, hash_matrix, i)
            movie_score.append([i, buffer_score])
    new_movie_score = sorted(movie_score, key=lambda x:x[1], reverse=True)
    recommendation_result = []
    for i in range(k):
        recommendation_result.append(movie_dict_reverse[new_movie_score[i][0]])
    return recommendation_result#推荐的是没看过的

def eval_all(test_data, train_data, tf_idf_matrix, movie_dict):
    #test_single = test_data[0]
    #user_single = test_single[0]
    #movie_single = test_single[1]
    #movie_rate_list_single = find_other_movie(user_single, movie_single, train_data)
    real_result = []
    pre_result = []
    #movie_dict:真实->矩阵
    movie_dict_reverse = {}#矩阵->真实
    for x in movie_dict:
        movie_dict_reverse[movie_dict[x]] = x
    for i in range(len(test_data)):
        movie_rate_list = find_other_movie(test_data[i][0], test_data[i][1], train_data)
        real_result.append(test_data[i][2])
        test_movie_juzhen_index = movie_dict[test_data[i][1]]
        pre_buffer = 0.0
        cosine_s_all = 0.0
        for k in range(len(movie_rate_list)):
            cosine_s = cosine_similar(tf_idf_matrix[movie_dict[movie_rate_list[k][0]]], tf_idf_matrix[test_movie_juzhen_index])
            if cosine_s<0:
                cosine_s=0
            cosine_s_all+=cosine_s
            #print(cosine_s)
            #print(movie_rate_list[k][1])
            pre_buffer+=cosine_s*movie_rate_list[k][1]
            #print(pre_buffer)
        pre_result.append(pre_buffer/cosine_s_all)

    loss = 0.0
    print(pre_result)
    print(real_result)
    for i in range(len(pre_result)):
        loss += (pre_result[i] - real_result[i])**2
    print('normol_loss', loss)

    return loss
    #print(movie_rate_list_single)
    #return

def eval_all_mini(test_data, train_data, matrix, movie_dict):
    #test_single = test_data[0]
    #user_single = test_single[0]
    #movie_single = test_single[1]
    #movie_rate_list_single = find_other_movie(user_single, movie_single, train_data)
    real_result = []
    pre_result = []
    #movie_dict:真实->矩阵
    movie_dict_reverse = {}#矩阵->真实
    for x in movie_dict:
        movie_dict_reverse[movie_dict[x]] = x
    for i in range(len(test_data)):
        movie_rate_list = find_other_movie(test_data[i][0], test_data[i][1], train_data)
        real_result.append(test_data[i][2])
        test_movie_juzhen_index = movie_dict[test_data[i][1]]
        pre_buffer = 0.0
        jaccard_s_all = 0.0
        for k in range(len(movie_rate_list)):
            jaccard_s = jaccard_sum_single(matrix[movie_dict[movie_rate_list[k][0]]], matrix[test_movie_juzhen_index])
            jaccard_s_all+=jaccard_s
            #print(cosine_s)
            #print(movie_rate_list[k][1])
            pre_buffer+=jaccard_s*movie_rate_list[k][1]
            #print(pre_buffer)
        pre_result.append(pre_buffer/jaccard_s_all)

    loss = 0.0
    print(pre_result)
    print(real_result)
    for i in range(len(pre_result)):
        loss += (pre_result[i] - real_result[i])**2
    print('mini_loss', loss)

    return loss

def jaccard_hash_sum_single(a, b):
    length = len(a)
    buffer = np.sum(a==b)
    return buffer/length

def eval_all_mini_hash(test_data, train_data, hash_matrix, movie_dict):
    #test_single = test_data[0]
    #user_single = test_single[0]
    #movie_single = test_single[1]
    #movie_rate_list_single = find_other_movie(user_single, movie_single, train_data)
    real_result = []
    pre_result = []
    #movie_dict:真实->矩阵
    movie_dict_reverse = {}#矩阵->真实
    for x in movie_dict:
        movie_dict_reverse[movie_dict[x]] = x
    for i in range(len(test_data)):
        movie_rate_list = find_other_movie(test_data[i][0], test_data[i][1], train_data)
        real_result.append(test_data[i][2])
        test_movie_juzhen_index = movie_dict[test_data[i][1]]
        pre_buffer = 0.0
        jaccard_s_all = 0.0
        for k in range(len(movie_rate_list)):
            jaccard_s = jaccard_hash_sum_single(hash_matrix[movie_dict[movie_rate_list[k][0]]], hash_matrix[test_movie_juzhen_index])
            jaccard_s_all+=jaccard_s
            #print(cosine_s)
            #print(movie_rate_list[k][1])
            pre_buffer+=jaccard_s*movie_rate_list[k][1]
            #print(pre_buffer)
        if jaccard_s_all==0:
            pre_result.append(2.5)
        else:
            pre_result.append(pre_buffer/jaccard_s_all)

    loss = 0.0
    print(pre_result)
    print(real_result)
    for i in range(len(pre_result)):
        loss += (pre_result[i] - real_result[i])**2
    print('mini_loss', loss)

    return loss

import pickle
import random

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

data_head, data = load_data('../datasets/movies.csv', 1)
print(data[:3])
print(len(data))
movie_id = {}#需要映射
movie_name = {}
genres = {}


for i in range(len(data)):
    movie_id[data[i][0]] = movie_id.get(data[i][0], 0)+1
    movie_name[data[i][1]] = movie_name.get(data[i][1], 0)+1
    for x in data[i][2]:
        genres[x] = genres.get(x, 0)+1

'''
for x in movie_id:
    if movie_id[x]!=1:
        print('偷袭', x, movie_id[x])

for x in movie_name:#名字又重复的，id无重复的
    if movie_name[x]!=1:
        print('偷袭', x)
'''

print(len(movie_id))
print(len(genres))
print(genres)

genres_hash = {}
indexs = 0
for x in genres:
    genres_hash[x] = indexs
    indexs+=1

movie_dict = {}
indexs2 = 0
for i in range(len(data)):
    movie_dict[eval(data[i][0])] = indexs2
    indexs2 += 1


matrix = matrix_make(data, genres_hash)#此矩阵极即为mini矩阵
a_mini_hash = mini_hash(5)
a_hash_matrix = a_mini_hash.hash_matrix(matrix)
#得到的是降维后的矩阵

#tf_idf_matrix = tf_idf_make(matrix, genres_hash, genres)
#print(tf_idf_matrix[0])

test_headers, test_data = load_data_eval('../datasets/test_set.csv', 1)
train_headers, train_data = load_data_eval('../datasets/train_set.csv', 1)

#eval_all(test_data, train_data, tf_idf_matrix, movie_dict)
#eval_all_mini(test_data, train_data, matrix, movie_dict)
import time
#begin = time.process_time()
#eval_all_mini_hash(test_data, train_data, a_hash_matrix, movie_dict)
#end = time.process_time()
#print('times:{}s'.format(end-begin))

#recommendation_single = recommendation_mini(547, 3, train_data, movie_dict, matrix)
#print(recommendation_single)#[3, 64, 68]

begin = time.process_time()
recommendation_single = recommendation_mini_hash(547, 3, train_data, movie_dict, a_hash_matrix)
end = time.process_time()
print('times:{}s'.format(end-begin))
print(recommendation_single)#[2394, 3429, 3725]

#print(matrix[1])
#print(genres_hash)

#matrix:9125x20
