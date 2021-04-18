def list_true_sub(alist):
    N = len(alist)
    res = []
    for i in range(2**N):
        combo = []
        for j in range(N):
            if(i>>j)&1:
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
        result.append(set(x))
    #print(result)
    return result


#a = frozenset([1, 2, 3])
#print(a)
#print(all_true_subset(a))
a = frozenset([1, 2])
b = frozenset([2, 1])
c = {a:2}
if a==b:
    print('hello')
print(c[b])