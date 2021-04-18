import csv

def Ck_make(data_set, froze_set_data):
    '''

    :param data_set: list for every item is a set
    :param froze_set_data:
    :return:
    '''
    result_set = []
    result_set_count = {}
    length = len(data_set)
    index = 0
    for i1 in range(len(data_set)):
        for i2 in range(i1+1, len(data_set)):
            result_set.append(set(data_set[i1]|data_set[i2]))
            #result_set_count[index] = 0
            #index+=1
    result_set = cut_repeat(result_set)
    for i in range(len(result_set)):
        result_set_count[i]=0
            #result_set_count[(i1, i2)] = 0
    #length = len(data_set)
    for i in range(len(froze_set_data)):
        if (i%100) ==0:
            print('times:%d'%(i))
        for j in range(len(result_set)):
            if result_set[j]<=froze_set_data[i]:
                result_set_count[j] += 1
                #result_set_count[(j//length, j-(j//length)*length)] += 1
    return result_set, result_set_count

def cut_repeat(Ck_list):
    '''new_Ck = set()
    for x in Ck_list:
        new_Ck.add(x)
    return new_Ck'''
    new_Ck = []
    cut_num=0
    length = len(Ck_list)
    for i in range(len(Ck_list)):
        if(i%100)==0:
            print("%d/%d"%(i, length))
        if Ck_list[i] not in new_Ck:
            new_Ck.append(Ck_list[i])
        else:
            cut_num+=1
    print('cut:%d', cut_num)
    return new_Ck

def filtr(Ck, Ck_count, all_line_num):
    new_Ck = []
    new_Ck_count = {}
    index = 0
    for i in range(len(Ck)):
        if Ck_count[i]>=0.005*all_line_num:
            new_Ck.append(Ck[i])
            new_Ck_count[index]=Ck_count[i]
            index+=1
    return new_Ck, new_Ck_count
    #for i in range(len(froze_set_data)):


data = []

item_count = {}

with open('Groceries.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    headers = next(csv_reader)
    for row in csv_reader:
        buffer = row[1].strip('{').strip('}').split(',')
        data.append(buffer)
        for x in buffer:
            item_count[x] = item_count.get(x, 0) + 1

print(data[:3])
set_data = []
froze_set_data = []
for line in data:
    set_data.append(set(line))
    froze_set_data.append(frozenset(line))
print(set_data[:3])
print(froze_set_data[:3])

all_name_set = set()
for x in set_data:
    all_name_set = all_name_set|x
print(all_name_set)#169个
all_line_num = len(set_data)
print(all_line_num)#9835
#print(len(item_count))#169
all_num = 0

C0 = []
for x in item_count:
    if item_count[x] >= 0.005*all_line_num:
        #print(x)
        C0.append(set([x]))

#print(item_count)
#print(len(C0))#120




'''
import pandas as pd
csv_data = pd.read_csv('Groceries.csv')
print(csv_data.shape)
print(csv_data[:])
'''



C1, C1_count=Ck_make(C0, froze_set_data)

print(len(C1))
print('C1_count')
print(C1_count)

C1v2, C1_count_v2 = filtr(C1, C1_count, all_line_num)
print(len(C1v2))
print(C1v2)

C2, C2_count=Ck_make(C1v2, froze_set_data)
C2v2, C2_count_v2 = filtr(C2, C2_count, all_line_num)
print('C2_num')
print(len(C2v2))
#print(C1_count)
