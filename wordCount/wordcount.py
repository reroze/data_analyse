import pickle

def map(filename):
    file = open(filename, 'r')
    word = {}
    for l in file.readlines():
        x = l.split(',')
        x = [y.strip() for y in x]
        x[-1] = x[-1].replace('\n', '')
        # print(x)
        for y in x:
            word[y] = word.get(y, 0) + 1

    #print(word)
    file.close()
    filew1 = open(filename+'.pkl', 'wb')

    pickle.dump(word, filew1)
    filew1.close()


if __name__ == '__main':
    '''file = open('source01', 'r')
    word = {}
    for l in file.readlines():
        x = l.split(',')
        x = [y.strip() for y in x]
        x[-1] = x[-1].replace('\n', '')
        #print(x)
        for y in x:
            word[y] = word.get(y, 0) + 1
    
    print(word)
    file.close()
    filew1 = open('word1.pkl', 'wb')
    
    pickle.dump(word, filew1)
    filew1.close()'''