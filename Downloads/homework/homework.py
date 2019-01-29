import random

def GCD(l):
    gcd = 1
    m = min(l)
    for i in range(2, m+1):
        cnt = 0
        for j in range(len(l)):
            if l[j] % i == 0:
                cnt+= 1
            if cnt == len(l):
                gcd = i
    return gcd

# def isSoroso(l):
#     m = max(l)
#     cnt = 0
#     res = False
#     # res = true
#     for i in range(2, m):
#         for k in range(len(l)):
#             if l[k] % i != 0:
#                 cnt = 1
#         if cnt == 0:
#             res = True
#             break
#     return res



def LCM(l):
    temp = l
    lcm = 1

    while True:
        flag = 0
        for i in range(temp):
            m = max(l)
            for j in range(m, 1, -1):
                if temp[i] % j == 0:
                    temp[i]/=j
                    lcm *= j

    return lcm

def main():
    a = [5, 10]
    # for i in range(10):
    #     a.append(random.randint(1,100))

    print(a)
    lcm_value = LCM(a)
    gcd_value = GCD(a)
    print('최소공배수:', lcm_value, ', 최대공약수:', gcd_value)
    print(isSoroso(a))
if __name__=='__main__':
    main()