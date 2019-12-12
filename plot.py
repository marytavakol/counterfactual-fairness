
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 14})
plt.figure()

auc = [0.6798, 0.6778]
auc_std = [0.0094, 0.0096]
prule = [0.8215, 0.8347]
prule_std = [0.0299, 0.0303]
replay = [1, 2]#, 3, 4, 5, 6, 7, 8, 9, 10]

axes = plt.gca()
axes.set_ylim([0, 1])

plt.plot(replay, auc, '-g', label='AUC')
plt.plot(replay, prule, '-b', label='Fairness')

# Plot std
auc_std_l = [auc_std[i] - auc_std[i] for i in range(len(auc))]
auc_std_u = [auc_std[i] + auc_std[i] for i in range(len(auc))]

p_std_l = [prule_std[i] - prule_std[i] for i in range(len(prule))]
p_std_u = [prule_std[i] + prule_std[i] for i in range(len(prule))]

plt.fill_between(replay, auc_std_l, auc_std_u, color='lightgreen')  # works!
plt.fill_between(replay, p_std_l, p_std_u, color='lightblue')  # works!

plt.legend(loc='upper right')

plt.xlabel('Replay Count')
plt.ylabel('Performance')


#plt.show()
plt.savefig("q")