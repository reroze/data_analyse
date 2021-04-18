import pickle

def reduce(filenames):
    words = {}
    for file_map_name in filenames:
        file_map = open(file_map_name, 'rb')
        map = pickle.load(file_map)
        for x in map:
            words[x] = words.get(x, 0) + map[x]
        file_map.close()
    file_reduce_name = filenames[0].replace('.pkl', '.reduce')
    file_reduce = open(file_reduce_name, 'wb')
    pickle.dump(words, file_reduce)
    file_reduce.close()


    return words


if __name__ == '__main__':

    '''file_map1 = open('word1.pkl', 'rb')
    file_map2 = open('word2.pkl', 'rb')

    word1 = pickle.load(file_map1)
    word2 = pickle.load(file_map2)

    file_map1.close()
    file_map2.close()

    words = word1

    for x in word2:
        words[x] = words.get(x, 0)+word2[x]

    print(words)'''

