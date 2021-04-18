import random

def random_hash(N):
    A = random.sample(range(1, 10*N), N)
    B = random.sample(range(1, 10*N), N)
    C = random.sample(range(2, 10*N), N)
    return A, B, C

import numpy as np

class mini_hash():
    def __init__(self, N):
        self.N = N
        self.As, self.Bs, self.Cs = random_hash(N)
        self.Hashs = []
        self.testHashs = [lambda x:(x+1)%5, lambda x:(3*x+1)%5]
        for i in range(N):
            hash_buffer = lambda x:(self.As[i]*x+self.Bs[i])%self.Cs
            self.Hashs.append(hash_buffer)

    def hash_single(self, i_index, j_index, hash_matrix):
        for i in range(self.N):
            hash_buffer = self.Hashs[i](j_index)
            if hash_buffer<hash_matrix[i_index][i]:
                hash_matrix[i_index][i]=hash_buffer

    def hash_matrix(self, matrix):
        '''

        :param matrix: (User,Matrix)
        :return:
        '''
        user_num = len(matrix)
        length = len(matrix[0])
        #hash_matrix = np.zeros(user_num, self.N)
        inf = 10*self.N+1
        hash_matrix = np.full([user_num, self.N], inf)
        for i in range(user_num):
            for j in range(length):
                if matrix[i][j]==1:
                    self.hash_single(i, j, hash_matrix)
        return hash_matrix

    def test_hash_single(self, i_index, j_index, hash_matrix):
        for i in range(self.N):
            hash_buffer = self.testHashs[i](j_index)
            if hash_buffer<hash_matrix[i_index][i]:
                hash_matrix[i_index][i]=hash_buffer

    def test_hash_matrix(self, matrix):
        '''

        :param matrix: (User,Matrix)
        :return:
        '''
        user_num = len(matrix)
        length = len(matrix[0])
        #hash_matrix = np.zeros(user_num, self.N)
        inf = 6
        hash_matrix = np.full([user_num, self.N], inf)
        for i in range(user_num):
            for j in range(length):
                if matrix[i][j]==1:
                    self.test_hash_single(i, j, hash_matrix)
        return hash_matrix






#hash1 = lambda x:(3*x+5)%8
#a = 3
#hash2 = hash1
#hash2 = lambda x:(4*x+7)%3
#print(hash2(a))


#matrix = np.zeros([10, 10])
#matrix = np.full([10, 10], -1)
#print(matrix)


#a = random.sample(range(2, 101), 10)
#print(a)

a_mini_hash = mini_hash(2)
a_matrix = np.array([
    [1,0,0,1,0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 1, 0]
])
a_hash_matrix = a_mini_hash.test_hash_matrix(a_matrix)
print(a_hash_matrix)
