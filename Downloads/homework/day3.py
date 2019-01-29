## 파일 입출력
'''
f = open("test.txt","r")
d = {}
for line in f:
    temp = line.split()
    for i in temp:
        print(i.upper(), end=' ')
    print()
f.close()
'''

f = open("score.txt","r")
num = list()
a = list()
b = list()
res1 = list()
grade = list()
for line in f:
    temp = line.split()
    num.append(temp[0])
    a.append(temp[1])
    b.append(temp[2])
    average = int(temp[1])*0.4+int(temp[2])*0.6
    if average >= 90:
        grade.append("A")
    elif average >= 80 and average <90:
        grade.append("B")
    elif average>= 70 and average <80:
        grade.append("C")
    elif average>=60 and average <70:
        grade.append("D")
    elif average <60:
        grade.append("F")
    res1.append(average)

f = open("report.txt","w")

for i in range(len(num)):
    data = num[i] +" "+ a[i]+" "+b[i]+" "+str(res1[i])+"("+grade[i]+")\n"
    f.write(data)
f.close()



## 함수
'''
def my_len(l):
    k = 0
    for i in l:
        k+=1

    return k

a = [5,5,6,7,8,3]
b = 'I am a boy.'

print(len(a), len(b))
print(my_len(a), my_len(b))
'''
'''
a = 1
b = 2

def add(x, y):
    return x+y

def sub(x, y):
    return x-y

def mul(x,y):
    return x*y

def div(x,y):
    return int(x/y)

c = add(a,b)
d = sub(a,b)
e = mul(a,b)
f = div(a,b)
print(c,d,e,f)
'''
'''
def factorial(n):
    k = 1
    for i in range(1,n+1,1):
        k= k*i
    return k

print(factorial(8))

def pow(n,a):
    k = n**a
    return k

print(pow(5,3))
print(pow(4,2))
print(pow(2,4))
'''
'''
def hour_min_sec(second):
    hour = int(second/3600)
    second = second - hour*3600
    m = int(second/60)
    second = second - m*60
    return (hour,m,second)

hour,m,sec = hour_min_sec(57894)

print("%d시간 %d분 %d초" %(hour,m,sec))
'''