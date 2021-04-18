import csv
import numpy

d = 0.85

def matrix_make(allnodes):
    length = len(allnodes)
    length -= 1
    nodes_buffer = allnodes[1:]
    matrix = numpy.zeros([length, length])
    for i in range(length):
        for source in nodes_buffer[i].source_nodes:
            matrix[i][source-1] = 1/nodes_buffer[source-1].des_nums
    return matrix


class node:
    def __init__(self, id, des_nodes, source_nodes, rank):
        self.id=id
        self.rank = rank
        self.des_nodes=des_nodes
        self.source_nodes = source_nodes
        self.source_nums = 0
        self.des_nums = 0

    def adddes(self, des):
        #print('hello')
        #print(des)
        if des not in self.des_nodes:
            self.des_nodes.append(des)
            #print(self.des_nodes)
            self.des_nums+=1

    def addsource(self, source):
        if source not in self.source_nodes:
            self.source_nodes.append(source)
            self.source_nums+=1


    '''def get_des(self):
        print(self.des_nodes)
        return self.des_nodes'''





data = []
with open('sent_receive.csv', 'r') as f:
    csv_file = csv.reader(f)
    headers = next(csv_file)

    for row in csv_file:
        data.append(row)



#print(data)
#print(headers)
#print(len(data))

#a = [1, 2, 3, 5]
#print(max(a))

max1 = []
max2 = []


for d in data:
    max1.append(int(d[1]))
    max2.append(int(d[2]))




#原来的1

nodes_dict = {}
for d in data:
    nodes_dict[int(d[1])]=1
    nodes_dict[int(d[2])]=1

print('length:', len(nodes_dict)) #180 1~462 一共180个节点

allnodes=[]
#第0个节点不存
for i in range(462):
    allnodes.append(node(i, [], [], 1/180))



for d in data:
    allnodes[int(d[1])].adddes(int(d[2]))
    allnodes[int(d[2])].addsource(int(d[1]))

#原来的1

'''
for i in range(462):
    if len(allnodes[i].des_nodes)!=0:
        print(allnodes[i].des_nodes)
        print(i)
        break'''
print('源结点最大值', max(max1))#461
print('源结点最小值', min(max1))#5
print('目的结点最大值', max(max2))#461
print('目的结点最小值', min(max2))#4



#原来的2

matrix = matrix_make(allnodes)
print(matrix)

count=0

for i in range(461):
    for j in range(461):
        if matrix[i][j]!=0:
            count+=1

#print(count, count/(461*461))

node_array = numpy.zeros([461, 1])
for i in range(461):
    node_array[i] = 1/180

old_array = node_array

#原来的2


def diedai(matrix, node_array):
    delta = 100
    count = 0
    new_node_array = []
    '''new_node_array = numpy.matmul(matrix, node_array)
    print('matrix.shape:{}'.format(matrix.shape), 'node_array.shape:{}'.format(node_array.shape))
    print('new_node_array.shape:', new_node_array.shape)
    delta = new_node_array - node_array
    delta = delta.reshape([1, -1])
    delta = delta ** 2
    delta = delta.sum()
    count += 1
    print(delta)'''
    while(delta>=1e-8):
        new_node_array = numpy.matmul(matrix, node_array)
        '''
        0.014194007466552212
        8.128944388747163e-06
        5.072752163103466e-12
        '''
        for i in range(len(new_node_array)):
            new_node_array[i][0] = 0.15/180 + 0.85*new_node_array[i][0]
            #0.01025517039458397
            #4.243359776828449e-06
            #1.9131860212008358e-12
        sum1 = 0
        for i in range(len(new_node_array)):
            sum1 += new_node_array[i][0]
        for i in range(len(new_node_array)):
            new_node_array[i][0] = new_node_array[i][0]/sum1
        #print('sum ', sum(new_node_array)[0])

        delta = new_node_array - node_array
        delta = delta.reshape([1, -1])
        delta = delta**2
        #delta = abs(delta)
        delta = delta.sum()
        count+=1
        node_array = new_node_array
        #print(delta)
        if(count%4==0):
            print(delta)
    print('count', count)

    return new_node_array

#原来的3

new_node_array = diedai(matrix, node_array)
#print(new_node_array)

#print('sum of array', sum(new_node_array)[0])

new_list = []

for x in new_node_array:
    new_list.append(x[0])

new_count = 0
for x in new_list:
    if x != 0:
        new_count+=1


import heapq

print('new_count', new_count)
print(new_list.index(max(new_list))+1)
print(max(new_list))
print(sum(new_list))

max_number = heapq.nlargest(10, new_list)
print(max_number)
for x in max_number:
    print(new_list.index(x)+1)
#原来的3


#测试用例
'''
test_list = [0, 0, 0, 0, 0]
for i in range(5):
    test_list[i] = node(i, [], [], 1/4)

test_data = [
    [1, 3],
    [3, 1],
    [1, 4],
    [2, 1],
    [4, 3],
    [1, 2],
    [4, 2],
    [2, 4]
]

for x in test_data:
    test_list[x[0]].adddes(x[1])
    test_list[x[1]].addsource(x[0])

matrix = matrix_make(test_list)
print(matrix)

old_node = numpy.zeros([4, 1])
for i in range(4):
    old_node[i][0]= 1/4

finally_node = deidai(matrix, old_node)
print(finally_node)
'''