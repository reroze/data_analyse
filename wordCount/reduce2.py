import pickle

def reducev2(filenames, shuffles, id):
    words = {}
    first_character = [chr(x + ord('a')) for x in range(26)]
    for file_map_name in filenames:
        file_map = open(file_map_name, 'rb')
        map = pickle.load(file_map)
        if id==0:
            for x in map:
                if x[0] not in first_character:
                    words[x] = words.get(x, 0) + map[x]
                elif x[0] in shuffles:
                    words[x] = words.get(x, 0) + map[x]
            file_map.close()
        else:
            for x in map:
                if x[0] in shuffles:
                    words[x] = words.get(x, 0) + map[x]
            file_map.close()
    filename_re = 'reduce'+str(id)+'.reducev2'
    #file_reduce_name = filenames[0].replace('.pkl', '.reducev2')
    file_reduce = open(filename_re, 'wb')
    pickle.dump(words, file_reduce)
    file_reduce.close()


    return words

import os

def reducev3(filename, shuffles, id):
    filename_re = 'reduce' + str(id) + '.reducev2'
    if not os.path.exists(filename_re):
        words = {}
    else:
        filename_re_op = open(filename_re, 'rb')
        words = pickle.load(filename_re_op)
        filename_re_op.close()
    first_character = [chr(x + ord('a')) for x in range(26)]

    file_map = open(filename, 'rb')
    map = pickle.load(file_map)
    if id==0:
        for x in map:
            if x[0] not in first_character:
                words[x] = words.get(x, 0) + map[x]
            elif x[0] in shuffles:
                words[x] = words.get(x, 0) + map[x]
        file_map.close()
    else:
        for x in map:
            if x[0] in shuffles:
                words[x] = words.get(x, 0) + map[x]
        file_map.close()

    #file_reduce_name = filenames[0].replace('.pkl', '.reducev2')
    file_reduce = open(filename_re, 'wb')
    pickle.dump(words, file_reduce)
    file_reduce.close()


    return words