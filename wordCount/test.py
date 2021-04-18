import pickle


file = open('source01.reduce_finaly', 'rb')
first_character = [chr(x+ord('a')) for x in range(26)]
print(first_character)
count = [0 for x in range(27)]
dict = {}


for i in range(26):
    dict[first_character[i]] = i


words = pickle.load(file)
file.close()
print(len(words))

for x in words:
    x = x.lower()
    #if dict.get(x[0], 26)==26:
        #print(x)
    count[dict.get(x[0], 26)] +=1

print(count)

print('count_sum:', sum(count))

cha_dcit = {}

for cha in dict:
    cha_dcit[cha] = count[dict.get(cha, 26)]

cha_dcit[26] = count[26]

print('cha_dict', cha_dcit)
print('count[-1]', count[-1])


count_also = [x for x in count]
print(count_also)

y=0
for x in count_also:
    y+=x

print(y)

shuffle1 = [0, 1, 2, 3, 4, 5, 26]
shuffle2 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]
shuffle3 = [15, 18, 19, 20, 21, 22, 23, 24, 25]

y1=0
y2=0
y3=0
for x in range(27):
    if x in shuffle1:
        y1+=count_also[x]
    if x in shuffle2:
        y2+=count_also[x]
    if x in shuffle3:
        y3+=count_also[x]

print('y1:%d y2:%d y3:%d' % (y1, y2, y3))

print('shuffle1:')
list1 = []
list2 = []
list3 = []
for x in shuffle1:
    if x!=26:
    #print(first_character[x])
        list1.append(first_character[x])
print(list1)

print('shuffle2:')
for x in shuffle2:
    #print(first_character[x])
    list2.append(first_character[x])
print(list2)

print('shuffle3:')
for x in shuffle3:
    list3.append(first_character[x])
    #print(first_character[x])
print(list3)