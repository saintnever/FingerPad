import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
def f(x, A, B):
    return (A-xmid)*x + B

df_model = pd.read_csv('correction_model.csv')
df_model['ft_x'] = df_model['fingertip'].apply(lambda x: x.lstrip('(').rstrip(')').split(',')[0])
df_model['ft_y'] = df_model['fingertip'].apply(lambda x: x.lstrip('(').rstrip(')').split(',')[1])
df_model['ph_x'] = df_model['ph'].apply(lambda x: x.lstrip('(').rstrip(')').split(',')[0])
df_model['ph_y'] = df_model['ph'].apply(lambda x: x.lstrip('(').rstrip(')').split(',')[1])

df_yeqn = list()
# df_model.to_csv('df_model.csv')
for i in range(6):
    df_yeqn.append(df_model[df_model['j'] == i])

xmid = 120

plt.figure()
xs = list()
ys = list()
for i in range(6):
    x = df_yeqn[i]['i'].astype(int).values
    y = df_yeqn[i]['ft_x'].astype(int).values
    for k in range(len(y)):
        y[k] = y[k] + (y[k]-xmid)*(k*10)/(65+k*10)
    # y = y - y.iloc[-1]

    plt.plot(x, y, marker='o', label='x_re = ' + str(i))
    # for j in range(len(y)):
    #     xs.append(i*10)
    #     ys.append(y.values[j])

plt.xlabel('x_re')
plt.ylabel('ft_x_pixel')
plt.legend()
plt.show()

plt.figure()
xs = list()
ys = list()
slopes = list()
for i in range(6):
    df_xeqn = df_model[df_model['i'] == i]
    x = df_xeqn['ph_x'].astype(int).values
    y = df_xeqn['h'].astype(int).values
    popt, pcov = curve_fit(f, x, y)
    slopes.append(popt[0]-xmid)
    print(popt[0], popt[1])
    for k in range(len(y)):
        y[k] = y[k] - 0.28 * (x[k] - x[0])
    # # y = y - y.iloc[-1]

    plt.plot(x, y, marker='o', label='i = ' + str(i))
    # for j in range(len(y)):
    #     xs.append(i*10)
    #     ys.append(y.values[j])
print(np.mean(slopes))
plt.xlabel('ph_x')
plt.ylabel('h')
plt.legend()
plt.show()

# curve fit to compensate for ft_x changes with distance (x_re)

# popt, pcov = curve_fit(f, xs, ys)
# plt.figure()
# plt.plot(xs, ys, marker='o')
# plt.show()
