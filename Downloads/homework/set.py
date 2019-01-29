import numpy as np
class Set:
    def __init__(self, member=[]):
        self.member = member
    def append(self, a):
        self.member.append(a)
    def delete(self, a):
        if a in self.member:
            self.member.remove(a)

    def union(self, s2):
        union_result = Set(self.member[:])
        for i in s2.member:
            if i not in union_result.member:
                union_result.append(i)
        return union_result
    def intersection(self, s2):
        inter_result = Set([])
        for i in self.member:
            if i in s2.member:
                inter_result.member.append(i)
        return inter_result

    def difference(self, s2):
        dif_result = Set(s2.member[:])
        for i in self.member:
            if i in dif_result.member:
                dif_result.member.remove(i)
        return dif_result

    def __add__(self, b):
        c = self.union(b)
        return c

    def __sub__(self, b):
        c = self.difference(b)
        return c

    def __truediv__(self, b):
        c = self.intersection(b)
        return c


a = Set([1,2,3])
b = Set([2,3,4])

c = a.union(b)
print(c.member)

d = a.difference(b)
print(d.member)

e = a.intersection(b)
print(e.member)

c = a + b
print(c.member)

d = a - b
print(d.member)

e = a / b
print(e.member)