def analyse(file):
    file_op = open(file, "r")
    number = 0
    for line in file_op.readlines():
        x = line.split(',')
        x = [y.strip() for y in x]
        x[-1] = x[-1].replace('\n', '')
        #for y in x:
        number += len(x)
    file_op.close()
    return number



file_names = ['source01', 'source02', 'source03', 'source04', 'source05', 'source06', 'source07', 'source08', 'source09']

for file in file_names:
    print('file', analyse(file))

import pickle

file_reduce_name = 'reduce0.reducev2_finaly'
file_reduce_name_op = open(file_reduce_name, 'rb')
finally_dict = pickle.load(file_reduce_name_op)
file_reduce_name_op.close()
#print(finally_dict)
number = 0
for x in finally_dict:
    number+=finally_dict[x]

print(number)



