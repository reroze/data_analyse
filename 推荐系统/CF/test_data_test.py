import csv

def load_data(file_csv, header):
    data = []
    with open(file_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        if header!=None:
            headers = next(csv_reader)
        for row in csv_reader:
            buffer = row
            for i in range(len(buffer)):
                buffer[i] = eval(buffer[i])
            data.append(buffer)
    if header==None:
        return data
    else:
        return (headers, data)

test_headers, test_data = load_data('../datasets/test_set.csv', 1)
train_headers, train_data = load_data('../datasets/train_set.csv', 1)

train_movie_dict = {}

for x in train_data:
    train_movie_dict[x[1]] = train_movie_dict.get(x[1], 0)+1

users = []
user_dict = {}
test_movie_dict = {}
for i in range(len(test_data)):
    users.append(test_data[i][0])
    user_dict[test_data[i][0]] = user_dict.get(test_data[i][0], 0)+1
    test_movie_dict[test_data[i][1]] = test_movie_dict.get(test_data[i][1], 0) + 1

users = sorted(users, reverse=True)
print(users)
print(len(user_dict))

for x in test_movie_dict:
    if x not in train_movie_dict:
        print('偷袭', x)

def looking(user, movie, data):
    for i in range(len(data)):
        if user == data[i][0]:
            if movie== data[i][1]:
                return True
    return False





