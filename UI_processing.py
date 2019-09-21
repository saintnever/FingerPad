import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib


df = pd.read_csv('UI_results.csv')

x = ['S0', 'S1', 'S2']
y = [df['button_s1'].mean(), df['button_s2'].mean(), df['button_s3'].mean()]
yerr = [df['button_s1'].std(), df['button_s2'].std(), df['button_s3'].std()]

print(stats.f_oneway(df['button_s1'].values, df['button_s2'].values, df['button_s3'].values))

font = {'size' : 18}
matplotlib.rc('font', **font)
fig = plt.figure()
ax = plt.errorbar(x, y, yerr=yerr, label='both limits (default)', linewidth=3.0, elinewidth=3, capsize=10, marker='o')
plt.ylim([0.5, 1.05])
plt.xlabel('Session')
plt.ylabel('Accuracy')
plt.grid()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
plt.show()

def process(index_ser, time_ser):
    index = np.array(index_ser)
    index = [int(x) for x in index[0].lstrip('[').rstrip(']').split(',')]
    time = np.array(time_ser)
    time = [float(x) for x in time[0].lstrip('[').rstrip(']').split(',')]

    slider_list = list()
    for i in range(len(index)):
        if index[i] >= 32:
            slider_list.append(time[i])

    return slider_list
# slider time process
st_s1 = process(df['slider_index_s1'], df['slider_time_s1'])
st_s2 = process(df['slider_index_s2'], df['slider_time_s2'])
st_s3 = process(df['slider_index_s3'], df['slider_time_s3'])
print(stats.f_oneway(st_s1, st_s2, st_s3))

print(np.mean(st_s1+st_s2+st_s3))
y_st = [np.mean(st_s1), np.mean(st_s2), np.mean(st_s3)]
yst_error = [np.std(st_s1), np.std(st_s2), np.std(st_s3)]

fig1 = plt.figure()
plt.errorbar(x, y_st, yerr=yst_error, label='both limits (default)', linewidth=3.0, elinewidth=3, capsize=10, marker='o')
plt.ylim([0, 15])
plt.xlabel('Session')
plt.ylabel('Time (s)')
plt.grid()
plt.show()

# between user plot
x = df['user'].values
df['button_avg'] = (df['button_s1'] + df['button_s2'] + df['button_s3']) / 3
df.plot(x='user',y='button_avg',marker='o')
plt.show()


