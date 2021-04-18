import pickle
filer1 = open('word1.pkl', 'rb')
word = pickle.load(filer1)
filer1.close()
print(word)