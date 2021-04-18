def get_max_k_index(alist, k):
    new_alist = sorted(alist, reverse=True)
    k_data = new_alist[:k]
    max_index = []
    for x in k_data:
        max_index.append(alist.index(x))
    return max_index

a = [1, 10, 2, 4, 2]
print(get_max_k_index(a, 3))