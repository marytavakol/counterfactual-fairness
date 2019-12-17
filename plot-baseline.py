
from matplotlib import pyplot as plt
import numpy


plt.rcParams.update({'font.size': 14})
plt.figure()

axes = plt.gca()
axes.set_ylim([0.2, 1])


gamma = numpy.arange(0, 1.1, 0.1)

#auc_base = [0.7415363937941511, 0.7089997678785358, 0.7162156268443916, 0.7082354424794358, 0.6937173250710403, 0.6747289385196366, 0.6769412064845574, 0.6647944069935403, 0.6538746223525479, 0.6409846508883306]
#prule_base = [0.6516839665136518, 0.9923344197523987, 0.9369926881443394, 0.9664895167860051, 0.8883266350992389, 0.8704406628352691, 0.8962967330865232, 0.811642457883561, 0.8664543560404764, 0.7713503742637706]

auc_base = [0.76459, 0.73693, 0.70803, 0.68885, 0.70746, 0.68270, 0.66710, 0.67273, 0.66940, 0.66701, 0.66013]
auc_std = [0.00156, 0.00106, 0.00143, 0.00670, 0.00706, 0.00674, 0.00529, 0.00502, 0.00251, 0.00152, 0.00290]
prule_base = [0.31482, 0.65303, 0.97301, 0.90292, 0.84709, 0.89057, 0.85211, 0.86681, 0.86766, 0.86963, 0.86460]
prule_std = [0.00545, 0.0090, 0.00675, 0.01936, 0.04353, 0.02041, 0.02619, 0.01587, 0.01713, 0.01737, 0.01280]

auc = [0.7125673764340564]*len(auc_base)
auc_s = [0.00653]*len(auc_base)
prule = [0.8854682239660849]*len(auc_base)
prule_s = [0.02245]*len(auc_base)


auc_std_l = [auc_base[i] - auc_std[i] for i in range(len(auc_base))]
auc_std_u = [auc_base[i] + auc_std[i] for i in range(len(auc_base))]

p_std_l = [prule_base[i] - prule_std[i] for i in range(len(prule_base))]
p_std_u = [prule_base[i] + prule_std[i] for i in range(len(prule_base))]

#plt.xscale("log")

plt.plot(gamma, auc, '-b', label='AUC Counterfactual')
plt.plot(gamma, auc_base, '--b', label='AUC Baseline')

plt.plot(gamma, prule, '-g', label='Fairness Counterfactual')
plt.plot(gamma, prule_base, '--g', label='Fairness Baseline')

auc_s_l = [auc[i] - auc_s[i] for i in range(len(auc))]
auc_s_u = [auc[i] + auc_s[i] for i in range(len(auc))]

p_s_l = [prule[i] - prule_s[i] for i in range(len(prule))]
p_s_u = [prule[i] + prule_s[i] for i in range(len(prule))]

plt.fill_between(gamma, auc_std_l, auc_std_u, color='lightblue')
plt.fill_between(gamma, p_std_l, p_std_u, color='lightgreen')

plt.fill_between(gamma, auc_s_l, auc_s_u, color='lightblue')
plt.fill_between(gamma, p_s_l, p_s_u, color='lightgreen')

plt.legend(loc='lower right')

plt.xticks(gamma)
plt.xlabel(r'Multiplicative loss factor ($\gamma$)')
plt.ylabel('Performance')

plt.savefig("baseline")
#plt.show()