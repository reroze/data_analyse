import csv

def list_true_sub(alist):
    N = len(alist)
    res = []
    for i in range(2**N):
        combo = []
        for j in range(N):
            if((i>>j)%2):
                combo.append(alist[j])
        if len(combo)!=0 and len(combo)!=N:
            res.append(combo)
    return res

def all_true_subset(single_set):
    slist = []
    for x in single_set:
        slist.append(x)
    result = []
    sub_list = list_true_sub(slist)
    for x in sub_list:
        result.append(frozenset(x))
    #print(result)
    return result

def Ck_make(set_dict, froze_set_data, k_times):
    '''

    :param data_set:
    :param froze_set_data:
    :return:
    '''
    print(len(set_dict))
    #result_set = []
    result_set_dict = {}
    #length = len(data_set)
    #index = 0
    result_all = frozenset()
    key_list = []
    for x in set_dict:
        key_list.append(x)
    for i in range(len(key_list)):
        for j in range(i+1, len(key_list)):
            if(len(key_list[i]|key_list[j])==k_times):
                result_set_dict[key_list[i]|key_list[j]] = 0
    '''for x in set_dict:
        for y in set_dict:
            if x!=y:
                result_set_dict[x|y]=0'''
    print(len(result_set_dict))
    #for i1 in range(len(data_set)):
        #for i2 in range(i1+1, len(data_set)):
            #result_set.append(set(data_set[i1]|data_set[i2]))
            #result_set_count[index] = 0
            #index+=1

    #for i in range(len(result_set)):
        #result_set_count[i]=0
            #result_set_count[(i1, i2)] = 0
    #length = len(data_set)
    for i in range(len(froze_set_data)):
        if (i%100) ==0:
            print('times:%d'%(i))
        '''for j in range(len(result_set)):
            if result_set[j]<=froze_set_data[i]:
                result_set_count[j] += 1'''
        for x in result_set_dict:
            if x<=froze_set_data[i]:
                result_set_dict[x]+=1
                #result_set_count[(j//length, j-(j//length)*length)] += 1
    return result_set_dict


def rule_make(rule, Ck, new_Ck, k_times):
    #rule = {}
    other = []
    if k_times==0:
        for new_single in new_Ck:
            true_sub_list = all_true_subset(new_single)
            for i in range(len(true_sub_list)):
                if new_Ck[new_single]/Ck[true_sub_list[i]]>0.5:
                    rule[true_sub_list[i]] = new_single - true_sub_list[i]
    if k_times==1:
        for new_single in new_Ck:
            true_sub_list = all_true_subset(new_single)
            for i in range(len(true_sub_list)):
                if len(true_sub_list[i])==1:
                    if rule.get(true_sub_list[i], 0)!=0:
                        other.append([true_sub_list[i], new_single])
                else:
                    if new_Ck[new_single]/Ck[true_sub_list[i]]>=0.5:
                        rule[true_sub_list[i]] = new_single - true_sub_list[i]
    if k_times==2:
        for new_single in new_Ck:
            true_sub_list = all_true_subset(new_single)
            for i in range(len(true_sub_list)):
                if len(true_sub_list[i])!=3:
                    if rule.get(true_sub_list[i], 0)!=0:
                        if rule.get(true_sub_list[i])!=new_single-true_sub_list[i]:
                            other.append([true_sub_list[i], new_single])
                else:
                    if new_Ck[new_single]/Ck[true_sub_list[i]]>=0.5:
                        rule[true_sub_list[i]] = new_single - true_sub_list[i]

    return rule, other

def rule_make_all(C_list, k_times):
    '''

    :param Ck_list:
    :param k_times: 假设为2
    :return:
    '''
    rule = []
    Ck = C_list[k_times]
    for new_single in Ck:
        true_sub_list = all_true_subset(new_single)
        #print(len(true_sub_list))
        for i in range(len(true_sub_list)):
            length = len(true_sub_list[i])
            if (Ck[new_single]/C_list[length-1].get(true_sub_list[i], 0.1))>=0.5:
                rule.append([true_sub_list[i], new_single - true_sub_list[i]])
    return rule




def filtr(Ck_dict, all_line_num):
    new_Ck_dict = {}
    for x in Ck_dict:
        if Ck_dict[x]>0.005*all_line_num:
            new_Ck_dict[x] = Ck_dict[x]
    return new_Ck_dict
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

C0 = {}
for x in item_count:
    if item_count[x] > 0.005*all_line_num:
        #print(x)
        C0[frozenset([x])] = item_count[x]

print(C0)
#print(item_count)
#print(len(C0))#120

C1 = Ck_make(C0, froze_set_data, 2)
C1v2 = filtr(C1, all_line_num)
print(len(C1v2))

#rule = {}
#rule, _ = rule_make(rule, C0, C1v2, 0)
#rule1 = rule.copy()

#print(C1v2[frozenset({'citrus fruit', 'margarine'})])
C2 = Ck_make(C1v2, froze_set_data, 3)
C2v2 = filtr(C2, all_line_num)
print(len(C2v2))

#rule, other2 = rule_make(rule, C1v2, C2v2, 1)
#rule2 = rule.copy()

C3 = Ck_make(C2v2, froze_set_data, 4)
C3v2 = filtr(C3, all_line_num)

#rule, other3 = rule_make(rule, C2v2, C3v2, 2)
#rule3 = rule

print(len(C3v2))
#print(len(C1))
#print(rule1)
#print('len(rule1):%d' %(len(rule1)))
#print('len(rule2):%d' %(len(rule2)))
#print(rule2)
#print(other2)

rule1 = rule_make_all([C0, C1v2], 1)
print(len(rule1))

rule2 = rule_make_all([C0, C1v2, C2v2], 2)
print(len(rule2))

rule3 = rule_make_all([C0, C1v2, C2v2, C3v2], 3)
print(len(rule3))

'''
print(len(rule3))
print(other3)
print(len(other3))
if len(other3)!=0:
    for i in range(len(other3)):
        if len(other3[i][0])==2:
            if C3v2[other3[i][1]]/C2v2.get(other3[i][0], 1e+9)>0.5:
                rule[other3[i][0]] = other3[i][1] - other3[i][0]
                print('add1')
        if len(other3[i][0])==1:
            if C3v2[other3[i][1]]/C1v2.get(other3[i][0], 1e+9)>0.5:
                rule[other3[i][0]] = other3[i][1] - other3[i][0]

print(len(rule))
'''
'''
import pandas as pd
csv_data = pd.read_csv('Groceries.csv')
print(csv_data.shape)
print(csv_data[:])
'''

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
print(len(C2v2))'''
#print(C1_count)
