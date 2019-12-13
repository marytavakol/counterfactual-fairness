
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 14})
plt.figure()

#### replay
# auc =  [0.6737904993013052, 0.6701476017547001, 0.676446175631618, 0.6901975812197177, 0.6546398880273443, 0.6862370978656884, 0.6729662858146777, 0.697232305929484, 0.6784168777740963, 0.6872998054589685]
# auc_std =  [0.015531087894857201, 0.014661968754472298, 0.014015621444697037, 0.012760656173736425, 0.015600811108760864, 0.0143275734103784, 0.015503600065377439, 0.013182113895133568, 0.014102680920931792, 0.013537946719120304]
# prule =  [0.7903955861604771, 0.7975228137434661, 0.7899308332338637, 0.8605956859557523, 0.8124348046910427, 0.8154760354906404, 0.7961256308796036, 0.8607612537711316, 0.8149388791359126, 0.8297971946579217]
# prule_std =  [0.04647873911141896, 0.051844898621588954, 0.043827389157313636, 0.043665190375053786, 0.04900302303736013, 0.03931782187488969, 0.05146058397702538, 0.046693222708503054, 0.04931840979128122, 0.038852757232902636]

#### k
auc =  [0.6838131361466288, 0.70119906989191, 0.7105314773043532, 0.7213030665449572, 0.7333887289351243, 0.7454904456695318, 0.7528323497799645, 0.7570673574259901, 0.7532409789673324, 0.753388963734112]
auc_std =  [0.012163614047327544, 0.007259778696733091, 0.00843412615109057, 0.0069395197004312285, 0.0033337377428920246, 0.004021719630946834, 0.00365671123163149, 0.002572285790962962, 0.002483440596615425, 0.002710645453346202]
prule =  [0.8384773843469537, 0.8011359250008072, 0.6991632507406921, 0.5975595784141872, 0.5582730232310311, 0.47654628246277025, 0.44205950448539155, 0.42479947501565096, 0.45880184359990156, 0.4593433111143358]
prule_std =  [0.0333204939251227, 0.027936342966192, 0.04087113621687171, 0.05093247091130581, 0.019559792637471724, 0.029978678406624232, 0.021032309629509336, 0.009176875377190396, 0.017584716284108345, 0.017326771667127823]

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

plt.xlabel('k')
#plt.xlabel('Replay Count')
plt.ylabel('Performance')


#plt.show()
plt.savefig("replay")