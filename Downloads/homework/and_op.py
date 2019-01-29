import numpy as np

x1 = np.array([0,0])
x2 = np.array([0,1])
x3 = np.array([1,0])
x4 = np.array([1,1])

W = np.array([[-1, -1] , [1, 1]])
b = np.array([2,0])
y1 = np.matmul(W, x1)+b
y2 = np.matmul(W, x2)+b
y3 = np.matmul(W, x3)+b
y4 = np.matmul(W, x4)+b

f1 = np.argmax(y1)
f2 = np.argmax(y2)
f3 = np.argmax(y3)
f4 = np.argmax(y4)
print(f1)
print(f2)
print(f3)
print(f4)
