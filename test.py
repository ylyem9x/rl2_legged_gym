import numpy as np

class a:
    class b:
        c = 1

A = a
A.b.c=2
B = a.b
B= A.b
print(B.c)
B.c = 3
print(A.b.c)