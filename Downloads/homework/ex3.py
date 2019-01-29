f = open("src.txt",'r')
b = open("dst.txt",'w')
res = list()

for line in f:
    temp = line.split()
    for i in temp:
        res.append(i)

print(res)

f.close()

f = open("dst.txt","w")

for i in range(len(res)):
    data = res[i]+" "
    f.write(data)
f.close()