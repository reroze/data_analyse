import pickle
'''
for i in range(9):
    file_name = 'source0' + str(i+1) + '.pkl'
    file_op = open(file_name, 'rb')
    data = pickle.load(file_op)
    file_save_name = 'map' + str(i+1) +'.txt'
    file_save_op = open(file_save_name, 'w')
    for x in data:
        file_save_op.write(x)
        file_save_op.write(':')
        file_save_op.write(str(data[x]))
        file_save_op.write('\n')
    file_save_op.close()
    file_op.close()'''

file_name = 'reduce0.reducev2_finaly'
file_op = open(file_name, 'rb')
data = pickle.load(file_op)
file_save_name = 'reduce0_v2_finaly'+'.txt'
file_save_op = open(file_save_name, 'w')
for x in data:
    file_save_op.write(x)
    file_save_op.write(':')
    file_save_op.write(str(data[x]))
    file_save_op.write('\n')
file_save_op.close()
file_op.close()