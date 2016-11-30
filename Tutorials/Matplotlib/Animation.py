__author__ = 'DavidFawcett'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 3
theta_max = np.linspace(0, np.pi, 19)
coefficients = [3, 4, 1]

fig = plt.figure()
ax = plt.axes(polar=True)
ax.set_theta_zero_location('N')
line, = ax.plot([], [], lw=2)
ax.set_rmax(0)
ax.set_rmin(-40)


def animate(i):
    i = i%19
    theta = np.linspace(-np.pi, np.pi, 361)
    pat = 0
    for index in range(len(coefficients)):
        pat = pat + coefficients[index]*np.cos(2*(index+1-1)*.5*np.pi*(np.cos(theta) - np.cos(theta_max[i])))
    pat = abs(pat)**2
    pat = pat/max(pat)
    pat = 10*np.log10(pat)
    line.set_data(theta, pat)
    return line,

anim = animation.FuncAnimation(fig, animate, interval=100)

plt.show()