
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 14})
plt.figure()

auc = [0.6798, 0.6778, 0.6811, 0.6824, 0.6836, 0.6809, 0.6811, 0.6812, 0.6819, 0.6824]
auc_std = [0.0094, 0.0096, 0.0092, 0.0091, 0.0089, 0.0092, 0.0092, 0.0091, 0.0089, 0.0087]
prule = [0.8215, 0.8347, 0.8403, 0.8464, 0.8422, 0.8426, 0.8413, 0.8429, 0.8490, 0.8484]
prule_std = [0.0299, 0.0303, 0.0301, 0.0288, 0.0292, 0.0290, 0.0288, 0.0285, 0.0275, 0.0275]
replay = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

axes = plt.gca()
axes.set_ylim([0, 1])

plt.plot(replay, auc, '-g', label='AUC')
plt.plot(replay, prule, '-b', label='Fairness')

# Plot std
auc_std_l = [auc[i] - auc_std[i] for i in range(len(auc))]
auc_std_u = [auc[i] + auc_std[i] for i in range(len(auc))]

p_std_l = [prule[i] - prule_std[i] for i in range(len(prule))]
p_std_u = [prule[i] + prule_std[i] for i in range(len(prule))]

plt.fill_between(replay, auc_std_l, auc_std_u, color='lightgreen')  # works!
plt.fill_between(replay, p_std_l, p_std_u, color='lightblue')  # works!

plt.legend(loc='lower right')

plt.xlabel('Replay Count')
plt.ylabel('Performance')


plt.show()
#plt.savefig("q")