import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.stats
import pandas as pd
import glob
import matplotlib as mpl
import statistics
import matplotlib.patches as mpatches


mpl.rcParams['font.family'] = 'Times New Roman'

path = r'/Users/anastashia/Desktop/PHYS8602/problem1/probelm1_1_data'
all_files = glob.glob(path + "/*.dat")

li = []

columns = ["t" ,"x" ,"v" ,"p" ,"E"]

for filename in all_files:
    df = pd.read_csv( filename , names = columns, delim_whitespace = True, index_col = 0, header=None)
    li.append(df)

frame = pd.concat(li, axis = 0, ignore_index=True)

def actual_shm_ef (x, k=1, x0=1):
    e = 0.5*k*(x-x0)**2
    f = -k * (x-x0)
    return e, f

def actual_shm_xv (t, A=50, omega=1, phi=0):
    x = 5 * np.cos(omega* t + phi)
    v = -5 * omega * np.sin(omega * t + phi)
    return x, v


def plot_results(t_max = 10 , dt = 0.1):
    t_points = np.arange(0, t_max + dt, dt)
    for phi in range(0, 50):
        for t in range (0, 10):
            x, v = actual_shm_xv(t, 1, 1, phi)
            plt.plot(x, v, 'r+')

for l in li:
    plt.plot(l.x, l.p, color='blue')
    plt.plot ()
    plot_results(t_max = 10, dt = 0.1)
    plt.xlabel('x(m)')
    plt.ylabel("p (kg*m/s)")
    plt.title('Phase-Space Plot, dt = 0.1 s')
    red_patch = mpatches.Patch(color='r', label = 'Expected Results')
    blue_patch = mpatches.Patch(color='b', label = "Euler's Method Results")
    plt.legend(handles=[red_patch, blue_patch], loc='best')
    

plt.show()

print(statistics.stdev(l.E)/np.sqrt(50))
print(statistics.stdev(l.x)/np.sqrt(50))

