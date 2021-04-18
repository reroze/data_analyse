list1 = [1, 2, 3, 4]

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

list2 = pair_list(list1)
print(list2)