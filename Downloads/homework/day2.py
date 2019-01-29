'''
d = {'kim':13 , 'park':2, 'youn':1}

for value in d.values():
    print(value)


for key in d.keys():
    print(d[key])


for k, v in d.items():
    print(v)
'''

'''
sentence = input("Enter a sentence:")
list = list(sentence)
d = {}
for i in range(0,len(list)):
    if list[i] in d.keys():
        d[list[i]] = d[list[i]]+1
    else:
        d[list[i]] = 1

print(d)
'''
tmp = input("Enter a sentence:")
c = {}
for i in tmp:
    if i in c:
        c[i]+= 1
    else:
        c[i] = 1

print(c)
'''
temp_list = sorted(d.items())
for k,v in temp_list:
    print(k,v)
'''