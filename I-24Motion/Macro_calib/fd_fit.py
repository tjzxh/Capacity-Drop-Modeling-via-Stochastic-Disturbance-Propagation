import numpy as np
from scipy.optimize import curve_fit
from numpy import sqrt, pi, exp, log
from lmfit import Model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# read the FD data from the fd_data: q (veh/h), k (veh/km), v (km/h)
data = pd.read_excel(open('../data/fd_data.xlsx', 'rb'), index_col=0, header=0)

flow, density, speed = data['q'].to_numpy(), data['k'].to_numpy(), data['v'].to_numpy()
mean, var, sample_num = [], [], []
# from the min density, max density, and interval, generate the density list
density_interval = 2
density_list = np.arange(min(density), max(density), density_interval)
# filter out the density that fall into the critical range (15~25)
density_list = [i for i in density_list if i <= 15 or 25 <= i]
real_density_list = []
for i in density_list:
    sub_index = np.where((density >= i) & (density < i + density_interval))
    if not len(flow[sub_index]):
        continue
    flowi, speedi, densityi = flow[sub_index], speed[sub_index], density[sub_index]
    mean.append(np.average(flowi))
    var.append(np.var(flowi))
    real_density_list.append(np.average(densityi))
    sample_num.append(len(flowi))
    # print(f"density {i} num {len(flowi)}")
mean, var, sample_weights = np.array(mean), np.array(var), np.array(sample_num) / np.sum(sample_num)


def mean_q(x, vm, a, b):
    L = 0.1
    lam2 = a * log(x) + b
    lam = lam2 * x * L
    u = lam * vm
    # mean_vk = 1 / lam - u * exp(-u) / lam / (1 - exp(-u))
    mean_qk = 1 / lam2 / L - x * vm * exp(-u) / (1 - exp(-u))
    return mean_qk


def var_q(x, L, vm, a, b):
    lam2 = a * log(x) + b
    lam = lam2 * x * L
    u = lam * vm
    # var_vk = 1 / (lam ** 2) - (u ** 2) * exp(-u) / (lam ** 2) / (1 - exp(-u)) ** 2
    # var_qk = 1 / (lam2 ** 2) / (L ** 2) - (x * vm) ** 2 * exp(-u) / (1 - exp(-u)) ** 2
    var_qk = 1 / (lam2 ** 2) / (L ** 2) - (x * vm) ** 2 / (exp(u) + exp(-u) - 2)
    return var_qk


# curve fitting with mean
mean_model = Model(mean_q)
mean_model.set_param_hint('vm', value=120, min=50, max=150)
mean_model.set_param_hint('a', value=0.025, min=0.01, max=0.1)
mean_model.set_param_hint('b', value=-0.1818, min=-0.3, max=-0.01)
mean_result = mean_model.fit(mean, x=real_density_list, weights=sample_weights)
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -0.1818      0.017    -10.618      0.000      -0.216      -0.148
# x1            -0.0395      0.005     -8.400      0.000      -0.049      -0.030
# ==============================================================================
# plot the mean and variance
# HighD mean R-squared:  0.8444045330838182
L1, vm1, a1, b1 = 0.1, mean_result.best_values['vm'], mean_result.best_values['a'], mean_result.best_values['b']
print(mean_result.fit_report())
# transform a1, b1 from km/h to m/s
a = 3.6 * a1
b = 3.6 * b1 + 3 * log(10) * a
# # transform a, b from m/s to km/h
# a, b = 0.142, 0.548
# a1 = a / 3.6
# b1 = (b - 3 * log(10) * a) / 3.6

# a, b = 0.10800000000205916, 0.40481025406079396 in m/s
# I-80: L1, vm1, a1, b1 = 0.1, 36.44563222392642, 0.07836849626568203, -0.3260000000069208
# HighD: L1, vm1, a1, b1 = 0.1, 117.8116120529737, 0.04159654859308057, -0.13757252193080938
# I-24 MOTION: L1, vm1, a1, b1 = 0.1, 102.6, 0.0395, -0.1818
# filter the flow that exceeds the y2 or lower than y1 with threshold of 100 for each density
lb = mean_q(density, vm1, a1, b1) - np.sqrt(var_q(density, L1, vm1, a1, b1))
ub = mean_q(density, vm1, a1, b1) + np.sqrt(var_q(density, L1, vm1, a1, b1))
# valid_ind = [i for i in range(len(flow)) if ub[i] + 100 > flow[i] > lb[i] - 100]
# density, flow = density[valid_ind], flow[valid_ind]

# calculate the R-squared of the fitted mean and variance
continuous_k = np.arange(0.1, 72, 0.1)
y1 = mean_q(continuous_k, vm1, a1, b1) - np.sqrt(var_q(continuous_k, L1, vm1, a1, b1))
y2 = mean_q(continuous_k, vm1, a1, b1) + np.sqrt(var_q(continuous_k, L1, vm1, a1, b1))
density_list = np.array(real_density_list)
avg_flow_predicted = mean_q(density_list, vm1, a1, b1)
std = np.sqrt(var_q(density_list, L1, vm1, a1, b1))
all_picp = []
# compute the empirical mean and variance
for ind, i in enumerate(density_list):
    valid_index = np.where(density == i)
    if not len(valid_index[0]):
        continue
    flowi, speedi = flow[valid_index], speed[valid_index]
    # compute picp, where the flow is in the range of [avg_flow_predicted-std, avg_flow_predicted+std]
    picp = np.sum((flowi >= avg_flow_predicted[ind] - std[ind]) & (flowi <= avg_flow_predicted[ind] + std[ind])) / len(
        flowi)
    # check if the picp is nan
    if np.isnan(picp):
        picp = 0
    all_picp.append(picp)
# compute the MAPE of the fitted mean against the empirical mean
mean_mape = np.sum(np.abs(mean - avg_flow_predicted) / mean) / len(mean)
# compue the PICP, where the flow is in the range of [mean-std, mean+std]
avg_picp = np.mean(all_picp)
print(f'mean MAPE: {mean_mape:.1%} and PICP: {avg_picp:.1%}')
# mean MAPE: 18.1% and PICP: 77.8%
# calculate the R-squared of the stacked fitted mean and variance
mean_R = r2_score(mean, mean_q(np.array(real_density_list), vm1, a1, b1))
# mean R-squared:  0.9663925542334426 unweighted: 0.8377550987236896
# var R-squared:  -4.997829213453557
# mean and var R-squared:  -4.0846356480694
print('mean R-squared: ', mean_R)
# plot the best fitted function
plt.rcParams['font.size'] = '30'
plt.rcParams["font.family"] = "Times New Roman"
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.fill_between(continuous_k, y1, y2, alpha=.1, linewidth=0, color='red', label='Estimated std')
ax0.plot(continuous_k, (y1 + y2) / 2, linewidth=2, color='red', label='Estimated mean')
# get the critical density within the range of [15, 25] points alpha=0.5
kc_range = np.where((density >= 15) & (density <= 25))
rest_range = np.where((density < 15) | (density > 25))
ax0.scatter(density[kc_range], flow[kc_range], c='b', s=30, alpha=0.1)
ax0.scatter(density[rest_range], flow[rest_range], c='b', label='I-24 MOTION data', s=30)
ax0.set_xlabel("$k$ (veh/km/lane)")
ax0.set_ylabel("$Q$ (veh/h/lane)")
# ax0.set_xlim([0, 50])
ax0.set_xticks(np.arange(0, 61, 10))
ax0.legend(loc="upper left", fontsize=20)
# ax0.set_aspect(1 / 60)
ax0.set_box_aspect(1)
plt.show()

# print the fitted capacity and the error percentage from real capacity (1776)
print(f"a, b, Capacity = {a}, {b}, {np.max((y1 + y2) / 2)}")
print(f"the error percentage: {100 * abs(np.max((y1 + y2) / 2) - 1776) / 1776}%")
# mean R-squared:  0.7434541618112233
# a, b, Capacity = 0.08707084260286868, 0.32955542137404753, 1731.2134133365917
# the error percentage: 2.521767267083803%
