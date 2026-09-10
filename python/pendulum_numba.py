import numpy as np
from math import sin,cos
import time
from numba import njit

@njit
def runge5(f, y0, h, N):
    y = np.zeros((len(y0), N+1))

    y[:,0] = y0
    for n in range(N):
        yn = y[:,n]
        k1 = f(yn)
        k2 = f(yn + h*k1/5)
        k3 = f(yn + h*2*k2/5)
        k4 = f(yn + h*9*k1/4 - h*5*k2 + h*15*k3/4)
        k5 = f(yn - h*63*k1/100 + h*9*k2/5 - h*13*k3/20 + h*2*k4/25)
        k6 = f(yn - h*6*k1/25 + h*4*k2/5 + h*2*k3/15 + h*8*k4/75)
        y[:,n+1] = yn + h*(17*k1 + 100*k3 + 2*k4 - 50*k5 + 75*k6) / 144

    return y

@njit
def fpend(y):
    th1 = y[0]
    th2 = y[1]
    om1 = y[2]
    om2 = y[3]

    s1, c1 = sin(th1), cos(th1)
    s2, c2 = sin(th2), cos(th2)
    sd = s1*c2 - c1*s2            # sin(th1-th2)
    cd = c1*c2 + s1*s2            # cos(th1-th2)
    s12 = sd*c2 - cd*s2           # sin(th1-2*th2)
    denom = 2 + 2*sd**2           # 3-cos(2*th1-2*th2)

    th1dot = om1
    th2dot = om2
    om1dot = (-3*s1 - s12 - 2*sd*(om2**2 + om1**2*cd)) / denom
    om2dot = 2*sd*(2*om1**2 + 2*c1 + om2**2*cd) / denom

    return np.array([th1dot, th2dot, om1dot, om2dot])

def test_timing():
    y0 = np.array([2.0,2.0,0.0,-1.0])
    h = 0.2
    T = 10000;

    runge5(fpend, y0, h, 10)        # compile before timing

    for iter in range(10):
        start = time.time()
        y = runge5(fpend, y0, h, int(round(T/h)))
        end = time.time()
        print(end - start)

    return y

y = test_timing()
