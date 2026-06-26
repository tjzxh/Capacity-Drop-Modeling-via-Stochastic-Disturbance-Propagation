import numpy as np
import matplotlib.pyplot as plt
from numpy import log
from sklearn import linear_model

#### unit: km, km/h
# l1 result
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -0.0123      0.001    -12.636      0.000      -0.014      -0.010
# x1            -0.0061      0.000    -20.849      0.000      -0.007      -0.006
# ==============================================================================
# l2 result
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -0.1818      0.017    -10.618      0.000      -0.216      -0.148
# x1            -0.0395      0.005     -8.400      0.000      -0.049      -0.030
# mean l3 0.000716178875759096
#### unit: m, m/s
# l1 result, R-squared: 0.804
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.4179      0.015     28.554      0.000       0.389       0.447
# x1            -0.0866      0.004    -22.346      0.000      -0.094      -0.079
# ==============================================================================
# l2 result
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.3279      0.059      5.522      0.000       0.211       0.445
# x1            -0.1422      0.017     -8.400      0.000      -0.176      -0.109
# ==============================================================================
# mean l3 0.009281678222649201
plt.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})
plt.rcParams["mathtext.fontset"] = "cm"
LaneID = 2
# [spacing, len(vs), res.x[0], res.x[1], res.x[2], res.fun]
results = np.loadtxt('../data/I24_lamda_lane' + str(LaneID) + '_m.txt')
# filter num > 100 and spacing > 0 and fun is finite
valid_ind = (results[:, 1] > 100) & (results[:, 0] > 15) & (results[:, -1] > 0)
spacing_l1, l1 = results[valid_ind, 0], results[valid_ind, 2]
valid_ind2 = (results[:, 1] > 100) & (results[:, 0] < 40) & (results[:, -1] > 0)
spacing, l2, l3, fun = results[valid_ind2, 0], results[valid_ind2, 3], results[valid_ind2, 4], results[:, 5]
mean_l3 = np.mean(l3[l3 > 0])
print(f"mean l3 {mean_l3}")
# regression of l2
# Coef interval with statsmodels
import statsmodels.api as sm

l1_res = sm.OLS(l1, sm.add_constant(log(spacing_l1))).fit()
l2_res = sm.OLS(l2, sm.add_constant(log(spacing))).fit()
# conf_interval = lr.conf_int(alpha=0.05)
print(l1_res.summary(), l2_res.summary())

# # Robustly fit linear model with RANSAC algorithm
# huber1 = linear_model.HuberRegressor().fit(log(spacing_l1).reshape(-1, 1), l1)
# huber2 = linear_model.HuberRegressor().fit(log(spacing).reshape(-1, 1), l2)
# # huber3 = linear_model.HuberRegressor().fit(spacing.reshape(-1, 1), l3)
# print(huber1.coef_, huber1.intercept_, huber2.coef_, huber2.intercept_)
# huber2.coef_[0] * log(spacing) + huber2.intercept_
# plot the regression
fig, ax = plt.subplots(1, 3, sharex=True)
ax[0].scatter(spacing_l1, l1, c='b', label='Calibrated value with I-24 data')
ax[0].plot(spacing_l1, l1_res.predict(sm.add_constant(log(spacing_l1))), 'r', label='Regression line')
ax[0].set_xlabel("$s$ (m)")
ax[0].set_ylabel(r"$\lambda_1$")
ax[1].scatter(spacing, l2, c='b', label='Calibrated value with I-24 data')
ax[1].plot(spacing, l2_res.predict(sm.add_constant(log(spacing))), 'r', label='Regression line')
# # plot the slope of l2_res * log(spacing) + 0.548 vs. log(spacing) 0.08707084260286868, 0.32955542137404753
ax[1].plot(spacing, -0.08707084260286868 * log(spacing) + 0.32955542137404753, 'g', label='Manual line')
ax[1].set_xlabel("$s$ (m)")
ax[1].set_ylabel(r"$\lambda_2$")
ax[2].scatter(spacing, l3, c='b', label='Calibrated value with I-24 data')
ax[2].set_xlabel("$s$ (m)")
ax[2].set_ylabel(r"$\lambda_3$")
ax[0].legend(loc="best", fontsize=18)
ax[1].legend(loc="best", fontsize=18)
ax[2].legend(loc="best", fontsize=18)
plt.show()
