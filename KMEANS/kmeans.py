import tkinter
import matplotlib

matplotlib.use('TkAgg')

import csv
import random

def Euler_dis(node1, node2):
    distance=0.0
    for i in range(len(node1)):
        distance += (node1[i]-node2[i])**2
    return distance

class node:
    def __init__(self, data):
        self.input = data[1:]
        self.true_label = data[0]
        self.pre_label = 0

    def predict(self, k_classify):
        single_loss = 0.0
        distance_buffer=0
        index_buf = 0
        for i in range(len(k_classify)):
            if i ==0 :
                distance_buffer = Euler_dis(self.input, k_classify[i][1:])
                self.pre_label = k_classify[i][0]
            else:
                distance = Euler_dis(self.input, k_classify[i][1:])
                if distance<distance_buffer:
                    distance_buffer = distance
                    self.pre_label = k_classify[i][0]
                    index_buf = i
        single_loss = Euler_dis(self.input, k_classify[index_buf][1:])
        return single_loss

def sum1(all_node, k_classify_node):
    all_loss = 0.0
    for i in range(len(all_node)):
        all_loss+=all_node[i].predict(k_classify_node)
    return all_loss

def list_add(list1, list2):
    list_buf = [0 for i in range(len(list1))]
    for i in range(len(list1)):
        list_buf[i] = list1[i]+list2[i]
    return list_buf

def classify_node_generate(all_node, k):
    classify_node = [[0 for i in range(14)] for j in range(k)]
    classify_num = [0, 0, 0]
    for i in range(len(classify_node)):
        classify_node[i][0] = i
    for i in range(len(all_node)):
        index = all_node[i].pre_label
        #print(index)
        classify_node[index][1:]=list_add(classify_node[index][1:], all_node[i].input)
        classify_num[index] += 1
    for i in range(k):
        for j in range(13):
            classify_node[i][1+j] /= classify_num[i]
    return classify_node

def random_init_3(data, k):
    init_node = random.sample(data, 1)
    init_data = init_node.copy()
    init_data[0][0] = 0
    #print(init_data)
    distance_max = 0
    #data_buf = 0
    node_index = 0
    for i in range(len(data)):
        #print(len(data[i][1:]))
        #print(len(init_data[0][1:]))
        distance_buf = Euler_dis(data[i][1:], init_data[0][1:])

        if distance_buf>distance_max:
            distance_max = distance_buf
            node_index = i
    data_buf = data[node_index].copy()
    data_buf[0] = 1
    init_data.append(data_buf)
    distance_max = 0
    node_index = 0
    for i in range(len(data)):
        distance1 = Euler_dis(data[i][1:], init_data[0][1:])
        distance2 = Euler_dis(data[i][1:], init_data[1][1:])
        distance_buf = distance1+distance2
        if distance1!=0 and distance2!=0:
            if distance_buf>distance_max:
                distance_max = distance_buf
                node_index = i
    data_buf = data[node_index].copy()
    data_buf[0] = 2
    init_data.append(data_buf)
    return init_data

def true_classify_node_generate(all_node, k):
    classify_node = [[0 for i in range(14)] for j in range(k)]
    classify_num = [0, 0, 0]
    for i in range(len(classify_node)):
        classify_node[i][0] = i
    for i in range(len(all_node)):
        index = all_node[i].true_label-1
        # print(index)
        classify_node[index][1:] = list_add(classify_node[index][1:], all_node[i].input)
        classify_num[index] += 1
    for i in range(k):
        for j in range(13):
            classify_node[i][1 + j] /= classify_num[i]
    return classify_node

def get_label_classify_node(pre_classify_node, true_nodes):
    distance_reocrd = 0
    type_record = 0
    for x in true_nodes:
        distance_buf = Euler_dis(pre_classify_node[1:], x[1:])
        if distance_reocrd==0:
            distance_reocrd = distance_buf
            pre_classify_node[0] = x[0]
        elif distance_buf<distance_reocrd:
            distance_reocrd = distance_buf
            pre_classify_node[0] = x[0]
    return pre_classify_node



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def druw(all_node, accur, loss):
    type_node = [[] for i in range(3)]
    pcy_array = np.zeros([len(all_node), 13])
    pca = PCA(n_components=2)
    for i in range(len(all_node)):
        pcy_array[i] = all_node[i].input
        #print(all_node[i].pre_label)
        #type_node[all_node[i].pre_label].append(all_node[i].input)
    new_pcy_array = pca.fit_transform(pcy_array)
    for i in range(len(all_node)):
        type_node[all_node[i].pre_label].append(pcy_array[i])
    x_scas = [[] for i in range(3)]
    y_scas = [[] for i in range(3)]
    for i in range(len(type_node)):
        for j in range(len(type_node[i])):
            x_scas[i].append(type_node[i][j][0])
            y_scas[i].append(type_node[i][j][1])
    plt.scatter(x_scas[0], y_scas[0], color='r')
    plt.scatter(x_scas[1], y_scas[1], color='b')
    plt.scatter(x_scas[2], y_scas[2], color='g')

    name = 'SSE={:.2f}, Acc={:.2f}'.format(loss, accur)
    plt.title(name)

    plt.show()


def print_k(alist, k):
    for i in range(min(len(alist), k)):
        print(alist[i])






data = []

with open('归一化数据.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        for i in range(len(row)):
            row[i] = eval(row[i])
        data.append(row)

#print(data[0])
print_k(data, 3)
print(len(data[0]))

node_num = len(data)
print(node_num)
all_node = []
for i in range(node_num):
    all_node.append(node(data[i]))

k_classify_node = random_init_3(data, 3)
#print(k_classify_node)
print_k(k_classify_node, 3)

count = 0
delta = 100.0
old_loss=0
loss = 0.0

while(delta>1e-9):

    print('old_loss', old_loss)
    loss=sum1(all_node, k_classify_node)
    print('new_loss', loss)
    delta = abs(loss-old_loss)
    k_classify_node=classify_node_generate(all_node, 3)
    old_loss = loss
    count+=1
    if(count%100)==0:
        print('count:', count)

print('delta:%f'%delta)
print(count)

true_classify_node = true_classify_node_generate(all_node, 3)
for i in range(len(k_classify_node)):
    k_classify_node[i] = get_label_classify_node(k_classify_node[i], true_classify_node)

for i in range(len(all_node)):
    all_node[i].pre_label = k_classify_node[all_node[i].pre_label][0]

valid_count = 0

for i in range(len(all_node)):
    if all_node[i].pre_label==all_node[i].true_label-1:
        valid_count+=1




print(k_classify_node)
accur = valid_count/len(all_node)
print('accuracy:{}%'.format(100*accur))

druw(all_node, accur, loss)



