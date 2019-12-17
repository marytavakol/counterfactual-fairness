
from matplotlib import pyplot as plt
import numpy


plt.rcParams.update({'font.size': 14})
plt.figure()

axes = plt.gca()
axes.set_ylim([0, 1])


gamma = numpy.arange(0.1, 1.1, 0.1)

auc_base = [0.7415363937941511, 0.7089997678785358, 0.7162156268443916, 0.7082354424794358, 0.6937173250710403, 0.6747289385196366, 0.6769412064845574, 0.6647944069935403, 0.6538746223525479, 0.6409846508883306]
prule_base = [0.6516839665136518, 0.9923344197523987, 0.9369926881443394, 0.9664895167860051, 0.8883266350992389, 0.8704406628352691, 0.8962967330865232, 0.811642457883561, 0.8664543560404764, 0.7713503742637706]

auc = [0.7125673764340564]*len(auc_base)
prule = [0.8854682239660849]*len(auc_base)


#plt.xscale("log")

plt.plot(gamma, auc, '-b', label='AUC')
plt.plot(gamma, auc_base, '--b', label='AUC Baseline')

plt.plot(gamma, prule, '-g', label='Fairness')
plt.plot(gamma, prule_base, '--g', label='Fairness Baseline')

# Plot std
# auc_std_l = [auc[i] - auc_std[i] for i in range(len(auc))]
# auc_std_u = [auc[i] + auc_std[i] for i in range(len(auc))]
#
# acc_std_l = [acc[i] - acc_std[i] for i in range(len(acc))]
# acc_std_u = [acc[i] + acc_std[i] for i in range(len(acc))]
#
# p_std_l = [prule[i] - prule_std[i] for i in range(len(prule))]
# p_std_u = [prule[i] + prule_std[i] for i in range(len(prule))]

# plt.fill_between(frac, auc_std_l, auc_std_u, color='lightblue')  # works!
# #plt.fill_between(frac, acc_std_l, acc_std_u, color='lightcoral')
# plt.fill_between(frac, p_std_l, p_std_u, color='lightgreen')  # works!

plt.legend(loc='lower left')

plt.xticks(gamma)
plt.xlabel('gamma')
plt.ylabel('Performance')

#plt.savefig("baseline")
plt.show()