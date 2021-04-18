import csv

def bit_search(bits, location):
    index1 = location//32
    index2 = location - index1*32
    buffer = 0x1
    buffer = buffer<<index2
    return bits[index1]&buffer

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

def bitmap_generate(basket, all_line_num):
    buffer = 0x1
    bit_vec1= 0b0
    bit_vec2 = 0b0
    for i in range(len(basket)):
        if i<32:
            if basket[i]>0.005*all_line_num:
                bit_vec1 = bit_vec1|buffer
            buffer = buffer<<1
            if i==31:
                buffer = 0x1
        else:
            if basket[i]>0.005*all_line_num:
                bit_vec2 = bit_vec2|buffer
            buffer = buffer<<1
    return [bit_vec1, bit_vec2]

def Ck_make(set_dict, froze_set_data, k_times, bit_map=None, basket=None):
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

    #k_numbers = {}
    index_bit = 0

    index_dict = {}

    for x in set_dict:
        key_list.append(x)
    for i in range(len(key_list)):
        for j in range(i+1, len(key_list)):
            if(len(key_list[i]|key_list[j])==k_times):
                if bit_map!=None:
                    basket[Hash1(i, j)]+=1
                result_set_dict[key_list[i]|key_list[j]] = 0
                

    if bit_map!=None:
        bit_map = bitmap_generate(basket, len(froze_set_data))
        print(bin(bit_map[0]), bin(bit_map[1]))
        print(basket)
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
        if (i%3000) ==0:
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

def Hash1(i, j):
    return (i*j)%64


data = []

item_count = {}

bit_map11 = 0b0
bit_map12 = 0b0
basket = [0 for i in range(64)]
one_numbers = {}
index1 = 0

all_basket_num_real= 0
all_basket_num_std = 0

def pair_list(alist):
    '''
    [0, 1, 2]
    :param min:0
    :param max:2
    :return:
    '''
    list1 = []
    for i in range(len(alist)):
        for j in range(i+1, len(alist)):
            list1.append([alist[i], alist[j]])
    return list1

with open('Groceries.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    headers = next(csv_reader)
    for row in csv_reader:
        buffer = row[1].strip('{').strip('}').split(',')
        data.append(buffer)
        index_buffer = index1
        list_buffer= []
        for x in buffer:
            item_count[x] = item_count.get(x, 0) + 1
            if x not in one_numbers:
                one_numbers[x] = index1
                index1+=1
            list_buffer.append(one_numbers[x])
        length = len(buffer)
        if(length>=2):
            all_basket_num_std+=length*(length-1)/2
            pairs = pair_list(list_buffer)
            #print(pairs)
            for pair in pairs:
                basket[Hash1(pair[0], pair[1])]+=1

print('index1:%d'%index1)







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



for i in range(len(basket)):

    all_basket_num_real+=basket[i]
    if i<32:
        bitbuf = 0x1 << i
        if basket[i]>0.005*all_line_num:
            bit_map11=bit_map11|bitbuf
    else:
        bitbuf = 0x1<<(i-32)
        if basket[i]>0.005*all_line_num:
            bit_map12=bit_map12|bitbuf

    #bitbuf=bitbuf<<i
    #print(bitbuf)
    print(basket[i])

print(bin(bit_map11), bin(bit_map12))
if all_basket_num_std==all_basket_num_real:
    print('yes', all_basket_num_std, all_basket_num_real)

bit_map21 = 0b0
bit_map22 = 0b0
basket2 = [0 for i in range(64)]

#主测试函数

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
#bit_map2 = 0b0
C2 = Ck_make(C1v2, froze_set_data, 3, [bit_map21, bit_map21], basket2)
C2v2 = filtr(C2, all_line_num)
print(len(C2v2))

#rule, other2 = rule_make(rule, C1v2, C2v2, 1)
#rule2 = rule.copy()
#主测试函数
bit_map31 = 0
bit_map32 = 0
basket3 = [0 for i in range(64)]

C3 = Ck_make(C2v2, froze_set_data, 4, [bit_map31, bit_map32], basket3)

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

#主测试函数
'''
rule1 = rule_make_all([C0, C1v2], 1)
print(len(rule1))

rule2 = rule_make_all([C0, C1v2, C2v2], 2)
print(len(rule2))

rule3 = rule_make_all([C0, C1v2, C2v2, C3v2], 3)
print(len(rule3))
'''


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
