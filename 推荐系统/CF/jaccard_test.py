import numpy as np
import scipy.spatial.distance as dist

a = np.array([1, 1, 1, 0, 0, 0])

b = np.array([1, 0, 1, 0, 0, 1])

c = np.array([a, b])
ds = dist.pdist([a, b], 'jaccard')

print(np.dot(a, b))
#print(np.sum(a)**0.5)
print(np.dot(a, b)/((np.sum(a)*np.sum(b))**0.5))
print(ds)