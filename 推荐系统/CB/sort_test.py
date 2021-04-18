

alist = [[1, 3], [2, 5], [3, 1], [4, 5], [5, 2]]

blist = sorted(alist, key=lambda x:x[1], reverse=True)
print(blist)