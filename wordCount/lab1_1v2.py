from wordcount import map
from reduce1 import reduce
import pickle
import random

import threading
from reduce2 import reducev3
'''threadLockmap1 = threading.Lock()
threadLockmap1.acquire()
threadLockmap2 = threading.Lock()
threadLockmap2.acquire()
threadLockmap3 = threading.Lock()
threadLockmap3.acquire()'''
threadLocks = [1, 1, 1, 1, 1, 1, 1, 1, 1]
reduceLocks = [1, 1, 1]



for i in range(len(threadLocks)):
    threadLocks[i] = threading.Lock()
    threadLocks[i].acquire()

for i in range(len(reduceLocks)):
    reduceLocks[i] = threading.Lock()
    reduceLocks[i].acquire()

def allreduce(filenames):
    words = {}
    for file_map_name in filenames:
        file_map = open(file_map_name, 'rb')
        map = pickle.load(file_map)
        for x in map:
            words[x] = words.get(x, 0) + map[x]
        file_map.close()
    file_reduce_name = filenames[0].replace('.reducev2', '.reducev2_finaly')
    file_reduce = open(file_reduce_name, 'wb')
    pickle.dump(words, file_reduce)
    file_reduce.close()

class MAPThread (threading.Thread):#ID 0~8
    def __init__(self, threadID, name, filename):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.filename = filename
        #self.counter = counter
        #self.type =
    def run(self):
        print ("开启线程： " + self.name)
        # 获取锁，用于线程同步
        #threadLock.acquire()
        #print_time(self.name, self.counter, 3)
        # 释放锁，开启下一个线程
        words = map(self.filename)
        threadLocks[self.threadID].release()
        print('map{} finished'.format(self.threadID))
        #return words

import os

class REDUCEThread (threading.Thread):#ID 0~8
    def __init__(self, threadID, name, filenames, shuffles):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.filenames = filenames
        self.shuffles = shuffles
        #self.counter = counter
        #self.type =
    def run(self):
        print ("开启线程： " + self.name)
        # 获取锁，用于线程同步
        #threadLock.acquire()
        #print_time(self.name, self.counter, 3)
        # 释放锁，开启下一个线程
        #words = map(self.filename)
        srandom = [i for i in range(9)]
        random.shuffle(srandom)
        for i in srandom:
            threadLocks[i].acquire()
            threadLocks[i].release()
            #threadLocks[i].release()
            #print("reduce执行" + self.name + ":%d" %(i))
            words = reducev3(self.filenames[i], self.shuffles, self.threadID)

            #print("reduce完成" + self.name + ":%d" % (i))
            #while 1:
                #if os.path.exists(self.filenames[i]):
                    #words = reducev3(self.filenames[i], self.shuffles, self.threadID)
                    #break


        reduceLocks[int(self.threadID/3)].release()

        print('reduce{} finished'.format(self.threadID))

class ALLREDUCEThread(threading.Thread):  # ID 0~8
    def __init__(self, threadID, name, filenames):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.filenames = filenames
        # self.counter = counter
        # self.type =

    def run(self):
        print("开启线程： " + self.name)
        # 获取锁，用于线程同步
        # threadLock.acquire()
        # print_time(self.name, self.counter, 3)
        # 释放锁，开启下一个线程
        # words = map(self.filename)
        for i in range(3):
            reduceLocks[i].acquire()
            print("i:%d finished" %(i))

        words = allreduce(self.filenames)
        print('allreduce{} finished'.format(self.threadID))

        #eturn words

mapthread0 = MAPThread(0, 'map0', 'source01')
mapthread1 = MAPThread(1, 'map1', 'source02')
mapthread2 = MAPThread(2, 'map2', 'source03')
mapthread3 = MAPThread(3, 'map3', 'source04')
mapthread4 = MAPThread(4, 'map4', 'source05')
mapthread5 = MAPThread(5, 'map5', 'source06')
mapthread6 = MAPThread(6, 'map6', 'source07')
mapthread7 = MAPThread(7, 'map7', 'source08')
mapthread8 = MAPThread(8, 'map8', 'source09')

filenames1 = ['source01.pkl', 'source02.pkl', 'source03.pkl', 'source04.pkl', 'source05.pkl', 'source06.pkl', 'source07.pkl', 'source08.pkl', 'source09.pkl']
#filenames2 = ['source04.pkl', 'source05.pkl', 'source06.pkl']
#filenames3 = ['source07.pkl', 'source08.pkl', 'source09.pkl']
reduce_filename = ['reduce0.reducev2', 'reduce3.reducev2', 'reduce6.reducev2']
shuffle1 = ['a', 'b', 'c', 'd', 'e', 'f']
shuffle2 = ['g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'q', 'r']
shuffle3 = ['p', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
reducethread0 = REDUCEThread(0, 'reduce0', filenames1, shuffle1)
reducethread1 = REDUCEThread(3, 'reduce3', filenames1, shuffle2)
reducethread2 = REDUCEThread(6, 'reduce6', filenames1, shuffle3)

allreducethread = ALLREDUCEThread(0, 'allreduce', reduce_filename)
mapthread0.start()

mapthread1.start()
#print('map1 finished')
mapthread2.start()
#print('map2 finished')
mapthread3.start()
mapthread4.start()
mapthread5.start()
mapthread6.start()
mapthread7.start()
mapthread8.start()
reducethread0.start()
reducethread1.start()
reducethread2.start()
allreducethread.start()
#print('reduce0 finished')
#print('主进程结束')







