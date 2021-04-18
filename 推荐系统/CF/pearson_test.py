import numpy as np

def expectation(line):
    buffer = 0.0
    for i in range(len(line)):
        buffer += line[i]
    buffer /= len(line)
    return buffer

def Cov(line_x, line_y):
    Ex = expectation(line_x)
    Ey = expectation(line_y)
    buffer = 0.0
    length = len(line_x)
    for i in range(length):
        buffer += (line_x[i]-Ex)*(line_y[i]-Ey)
    buffer /= length
    return buffer

def standard_Devia(line, E):
    buffer = 0.0
    for i in range(len(line)):
        buffer += (line[i]-E)**2
    buffer /= len(line)
    buffer = buffer ** 0.5
    return buffer

def pearson_sum(matrix, index):
    number = len(matrix)
    result = []
    for i in range(len(matrix)):
        buffer = Cov(matrix[i], matrix[index])/(standard_Devia(matrix[i],
        expectation(matrix[i]))*standard_Devia(matrix[index], expectation(matrix[index])))
        result.append(buffer)
    return result

list1 = [[5.0, 3.0, 2.5],
                   [2.0, 2.5, 5.0]]

matrix = np.array(list1)
print(matrix)

pearson = pearson_sum(matrix, 0)
print(pearson)
