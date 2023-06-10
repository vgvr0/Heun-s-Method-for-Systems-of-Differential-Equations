import numpy as np
import matplotlib.pyplot as plt

def heun_method(f, y0, t0, h, num_steps):
    num_eqs = len(y0)
    t = np.zeros(num_steps+1)
    y = np.zeros((num_eqs, num_steps+1))
    y[:, 0] = y0
    
    for i in range(num_steps):
        t[i+1] = t[i] + h
        
        k1 = f(t[i], y[:, i])
        k2 = f(t[i] + h/2, y[:, i] + h/2 * k1)
        
        y[:, i+1] = y[:, i] + h * (k1 + k2) / 2
    
    return t, y

# Ejemplo de sistema de ecuaciones diferenciales: dy1/dt = y2, dy2/dt = -y1
def f(t, y):
    return np.array([y[1], -y[0]])

# Condiciones iniciales y parámetros
y0 = np.array([1, 0])
t0 = 0
h = 0.1
num_steps = 100

# Aplica el método de Heun
t, y = heun_method(f, y0, t0, h, num_steps)

# Grafica las soluciones
plt.plot(t, y[0, :], label='y1(t)')
plt.plot(t, y[1, :], label='y2(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()
