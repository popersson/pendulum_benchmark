import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"   # single threaded, as the others

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import time

@partial(jax.jit, static_argnums=(0,3))
def runge5(f, y0, h, N):
    y = jnp.zeros((len(y0), N+1)).at[:,0].set(y0)

    def step(n, y):
        yn = y[:,n]
        k1 = f(yn)
        k2 = f(yn + h*k1/5)
        k3 = f(yn + h*2*k2/5)
        k4 = f(yn + h*9*k1/4 - h*5*k2 + h*15*k3/4)
        k5 = f(yn - h*63*k1/100 + h*9*k2/5 - h*13*k3/20 + h*2*k4/25)
        k6 = f(yn - h*6*k1/25 + h*4*k2/5 + h*2*k3/15 + h*8*k4/75)
        return y.at[:,n+1].set(yn + h*(17*k1 + 100*k3 + 2*k4 - 50*k5 + 75*k6) / 144)

    return jax.lax.fori_loop(0, N, step, y)

def fpend(y):
    th1, th2, om1, om2 = y

    s1, c1 = jnp.sin(th1), jnp.cos(th1)
    s2, c2 = jnp.sin(th2), jnp.cos(th2)
    sd = s1*c2 - c1*s2            # sin(th1-th2)
    cd = c1*c2 + s1*s2            # cos(th1-th2)
    s12 = sd*c2 - cd*s2           # sin(th1-2*th2)
    denom = 2 + 2*sd**2           # 3-cos(2*th1-2*th2)

    th1dot = om1
    th2dot = om2
    om1dot = (-3*s1 - s12 - 2*sd*(om2**2 + om1**2*cd)) / denom
    om2dot = 2*sd*(2*om1**2 + 2*c1 + om2**2*cd) / denom

    return jnp.array([th1dot, th2dot, om1dot, om2dot])

def test_timing():
    y0 = jnp.array([2.0,2.0,0.0,-1.0])
    h = 0.2
    T = 10000;
    N = int(round(T/h))

    runge5(fpend, y0, h, N).block_until_ready()   # compile before timing

    for iter in range(10):
        start = time.time()
        y = runge5(fpend, y0, h, N).block_until_ready()
        end = time.time()
        print(end - start)

    return y

y = test_timing()
