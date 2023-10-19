import matplotlib.pyplot as plt
import numpy as np

x = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
min_t = [10, 12, 8, 15, 9]
max_t = [25, 28, 22, 30, 27]
avg_t = [(min_t + max_t) / 2 for min_t, max_t in zip(min_t, max_t)]

"""
plt.plot(x,max_t,'-.Dr',markersize=7, label='max')
plt.plot(x,min_t,'-.*g',markersize=15, label='min')
plt.plot(avg_t,'-.3b',markersize=10, label='mavg')
plt.xlabel('Day')
plt.ylabel('temp')
plt.title('day-temp')
plt.legend(shadow=True,fontsize='small')
plt.grid()
"""
xpos=np.arange(len(x))
plt.yticks(xpos,x)
plt.xlabel('Day')
plt.ylabel('temp')
plt.title('day-temp')
plt.barh(xpos-0.2,max_t,label="max temp",)
plt.barh(xpos,avg_t,label="avg temp",)
plt.barh(xpos+0.2,min_t,label="min temp",)
plt.legend()

plt.show()
