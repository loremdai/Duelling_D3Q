from collections import deque

q = deque(maxlen=2)

a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
q.append(a)
q.append(b)
q.append(c)
print(q)
print(len(q))
