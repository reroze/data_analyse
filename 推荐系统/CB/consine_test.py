import numpy as np


def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def cosine_myself(x, y):
    fenzi = 0.0
    x_data = 0.0
    y_data = 0.0
    for i in range(len(x)):
        fenzi += x[i]*y[i]
        x_data += x[i]**2
        y_data += y[i]**2

    x_data = x_data**0.5
    y_data = y_data**0.5
    return fenzi/(x_data*y_data)

a = np.array([1, 1, 2, 2, 3, 4])
b = np.array([1, 1, 2, 2, 3, 8])
print(cosine_similarity(a, b))
print(cosine_myself(a, b))