import numpy as np
from matplotlib import pyplot as plt

x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-10, 10, 100)
x3 = np.linspace(0, 1, 100)

y = np.exp(x1)
plt.plot(x1, y)
plt.title('$\exp (x)$')
plt.show()

y = 1 / (1 + np.exp(-x2) + 1e-10) # to avoid zerodivision
plt.plot(x2, y)
plt.title('Sigmoid Function:$\\frac{1}{1 + \exp (-z)}$')
plt.show()

y = np.log(x2)
plt.plot(x2, y)
plt.title('$\ln (x)$')
plt.show()

y = -np.log(x3)
plt.plot(x3, y, label='$y=1$')
y = -np.log(1 - x3)
plt.plot(x3, y, label='$y=0$')
plt.legend()
plt.title('Cross-Entropy Loss:$-y\ln(f(x))-(1-y)\ln (1-f(x))$')
plt.show()
