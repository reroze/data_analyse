import numpy as np

a = np.zeros([10])
print(a)
a[2] = 2
a[3] = 5
print(a)
b = a.copy()
print(b)

a = a/sum(a)
print(a)


'''
sum(c)和print(np.sum(c, axis=0))
[[0. 0. 2. 5. 0. 0. 0. 0. 0. 0.]
 [0. 0. 2. 5. 0. 0. 0. 0. 0. 0.]]->[ 0.  0.  4. 10.  0.  0.  0.  0.  0.  0.]

print(np.sum(c, axis=1))
[7. 7.]
14.0
print(np.sum(c))

'''