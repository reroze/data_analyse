from wordcount import map
from reduce1 import reduce
import pickle

import threading
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
    file_reduce_name = filenames[0].replace('.reduce', '.reduce_finaly')
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

class REDUCEThread (threading.Thread):#ID 0~8
    def __init__(self, threadID, name, filenames):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.filenames = filenames
        #self.counter = counter
        #self.type =
    def run(self):
        print ("开启线程： " + self.name)
        # 获取锁，用于线程同步
        #threadLock.acquire()
        #print_time(self.name, self.counter, 3)
        # 释放锁，开启下一个线程
        #words = map(self.filename)
        for i in range(3):
            threadLocks[self.threadID+i].acquire()
        reduceLocks[int(self.threadID/3)].release()
        words = reduce(self.filenames)
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

filenames1 = ['source01.pkl', 'source02.pkl', 'source03.pkl']
filenames2 = ['source04.pkl', 'source05.pkl', 'source06.pkl']
filenames3 = ['source07.pkl', 'source08.pkl', 'source09.pkl']
reduce_filename = ['source01.reduce', 'source04.reduce', 'source07.reduce']
reducethread0 = REDUCEThread(0, 'reduce0', filenames1)
reducethread1 = REDUCEThread(3, 'reduce3', filenames2)
reducethread2 = REDUCEThread(6, 'reduce6', filenames3)

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







