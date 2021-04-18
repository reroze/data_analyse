import csv

list = []

stopwords = ['', ',']

with open('datasets/datasets/Emails.csv', 'r') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)

    for row in f_csv:
        #print(row)
        if row[3] not in stopwords and row[4] not in stopwords:
            list.append([row[3], row[4]])



print(headers)#metadataTo:3  MetadataFrom:4
print(list)
print(len(list))
