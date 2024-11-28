class A:
    def __init__(self):
        pass
    def call_func(self, func):
        func(self.print1)
    def print1(self,num):
        print(num)

def func(Afun):
    Afun(2)

a = A()
a.call_func(func = func)