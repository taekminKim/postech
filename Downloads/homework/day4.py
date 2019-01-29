'''
l = [3,5,9,1,2]
def get_min_max(l):
    mini = min(l)
    maxi = max(l)
    l.remove(mini)
    l.remove(maxi)
    return mini, maxi

(min_val, max_val) = get_min_max(l)
print(min_val)
print(max_val)
print(l)
'''
'''
def sparseVDP(v1, v2):
    res = 0
    for i1,j1 in v1.items():
        for i2,j2 in v2.items():
            if i1==i2:
                res+= j1*j2
    return res
'''
'''
def sparseVDP(a,b):
    res = 0
    for k1 in a:
        if k1 in b:
            res+=a[k1]*b[k1]
        else:
            res+=0
    return res
'''
'''
def sparseVDP(a,b):
    res = 0
    for k1 in a:
        try:
            res+=a[k1] * b[k1]
        except KeyError:
            print("error")
            res+=0
    return res

v1 = {'c': 5}
v2 = {'a': 2 , 'b':1}
rr = sparseVDP(v1, v2)
print(rr)
'''
'''
def letter_dict(str):
    d={}
    for i in str:
        if i in d:
            d[i]+=1
        else:
            d[i]= 1
    return d

def max_letter(dict):
    v = max(dict.values())
    for k,val in dict.items():
        if val == v:
            res = k
    return res

a = letter_dict('red apple')
print(a)
print(max_letter(a))

def comb_dict(dict1, dict2):
    d={}

    for i in dict1:
        if i in d:
            d[i]+= dict1[i]
        else:
            d[i] = 1

    for j in dict2:
        if j in d:
            d[j]+= dict2[j]
        else:
            d[j] = 1
    return d

b = letter_dict('yellow banana')
c = comb_dict(a, b)

print(c)
print(max_letter(c))
'''
class Account():

    # number = 'OOOO-OOO-OOOOOO'
    # balance = 0
    # rate = 1.0


    # def __init__(self): Default 생성자
    def __init__(self, num='OOO-OOO-OOOOO', amnt=0, rate=1.0):
        self.number = num
        self.balance = amnt
        self.rate = rate

# 계좌의 기능(Method)
    def deposit(self, money): #입금
       self.balance += money
    def withdraw(self, money): #인출
      self.balance -= money
    def obtain_interest(self): #이자 획득
      self.balance += self.balance*(self.rate/100)



class MinBalanceAccount(Account): #Account 클래스를 상속받음
    def __init__(self, min_balance, num='OOOO-OOO-OOOOO', amnt=0, rate=1.0):
        Account.__init__(self, num=num, amnt=amnt, rate=rate)
        self.minimum_balance = min_balance
        self.bonus_rate = 1.0

    def withdraw(self, amnt): # 인출 기능 수정
        if self.balance - amnt < self.minimum_balance:
            print('Sorry, minimum balance must be maintained')
        else:
            Account.withdraw(self, amnt)
    def obtain_interest(self): #보너스 이율 반영
        self.balance += (self.balance)*((self.rate+self.bonus_rate)/100.0)

    def __add__(self, another):  # 두 계좌 통합 기능
        new_acc = Account(amnt=self.balance + another.balance, rate=self.rate)
        return new_acc


acc1 = Account()
acc2 = Account()
acc3 = Account()
acc1.deposit(500)
acc2.deposit(1000)
print(acc1.balance)
print(acc2.balance)
print(acc3.balance)