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

import random

loss = 0.0
#buffer = random.uniform(0.0, 5.0)
#print(buffer)

for i in range(len(test_data)):
    buffer = random.uniform(0.0, 5.0)
    #print(buffer)
    loss += (buffer-test_data[i][2])**2
    #print(data[i][3])
    #print(loss)

print(loss)
