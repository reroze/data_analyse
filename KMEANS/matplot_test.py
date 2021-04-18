import tkinter
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 100)
y1 = 3 * x + 4
y2 = x ** 2

plt.plot(x, y1)
plt.plot(x, y2)

plt.show()