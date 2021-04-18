'''a = [1, 2, 3]
b = set(a)
print(b)
for i in range(len(a)):
    print(a[i])
print(a[2])
'''
'''
c = {1,2,3}
d = {2,3,4}
x = set(c|d)
y = x
print(x)
print(y<=x)
'''
c = [
    {'hello', 'buffer'},
    {'hello', 'good'},
    {'hello', 'buffer'}
]

newx = set({'hello', 'good'})

if newx in c:
    print('hello')
a = frozenset([1, 2])
c = frozenset()

print(c)
'''
buf = 'string'

set_buf = set(buf)
print(set_buf)
buf += 'x'
print(buf)
print(set_buf)
'''

dict1 = {1:"1", 2:"2", 3:"3"}
for x in dict1:
    for y in dict1:
        print(x, " ", y)

a = frozenset(['hello', 'world'])
#print(a)
b = str(a)
print(b)
