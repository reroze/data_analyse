import pickle

Hashs = [lambda x:(x+1)%5, lambda x:(3*x+1)%5]
test_file = 'pickle_test'
test_file_op = open(test_file, "wb")
pickle.dump(Hashs, test_file_op)
test_file_op.close()