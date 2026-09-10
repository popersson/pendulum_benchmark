import numpy as np
from math import sin,cos
import time

class State(tuple):
    """Four floats with element-wise arithmetic, in the spirit of an SVector."""
    __slots__ = ()

    def __add__(self, y):
        return State((self[0]+y[0], self[1]+y[1], self[2]+y[2], self[3]+y[3]))

    def __sub__(self, y):
        return State((self[0]-y[0], self[1]-y[1], self[2]-y[2], self[3]-y[3]))

    def __rmul__(self, a):
        return State((a*self[0], a*self[1], a*self[2], a*self[3]))

    def __truediv__(self, a):
        return State((self[0]/a, self[1]/a, self[2]/a, self[3]/a))

    __mul__ = __rmul__

def runge5(f, y0, h, N):
    y = np.zeros((np.size(y0), N+1))

    yn = State(y0)
    y[:,0] = yn
    for n in range(N):
        k1 = f(yn)
        k2 = f(yn + h*k1/5)
        k3 = f(yn + h*2*k2/5)
        k4 = f(yn + h*9*k1/4 - h*5*k2 + h*15*k3/4)
        k5 = f(yn - h*63*k1/100 + h*9*k2/5 - h*13*k3/20 + h*2*k4/25)
        k6 = f(yn - h*6*k1/25 + h*4*k2/5 + h*2*k3/15 + h*8*k4/75)
        yn = yn + h*(17*k1 + 100*k3 + 2*k4 - 50*k5 + 75*k6) / 144
        y[:,n+1] = yn

    return y

def fpend(y):
    th1, th2, om1, om2 = y

    s1, c1 = sin(th1), cos(th1)   # no sincos in math, so four calls
    s2, c2 = sin(th2), cos(th2)
    sd = s1*c2 - c1*s2            # sin(th1-th2)
    cd = c1*c2 + s1*s2            # cos(th1-th2)
    s12 = sd*c2 - cd*s2           # sin(th1-2*th2)
    denom = 2 + 2*sd**2           # 3-cos(2*th1-2*th2)

    th1dot = om1
    th2dot = om2
    om1dot = (-3*s1 - s12 - 2*sd*(om2**2 + om1**2*cd)) / denom
    om2dot = 2*sd*(2*om1**2 + 2*c1 + om2**2*cd) / denom

    return State((th1dot, th2dot, om1dot, om2dot))

def test_timing():
    y0 = [2,2,0,-1]
    h = 0.2
    T = 10000;

    for iter in range(10):
        start = time.time()
        y = runge5(fpend, y0, h, int(round(T/h)))
        end = time.time()
        print(end - start)

    return y

y = test_timing()
