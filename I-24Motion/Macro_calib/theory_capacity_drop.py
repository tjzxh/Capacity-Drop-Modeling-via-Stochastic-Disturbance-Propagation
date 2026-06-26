import numpy as np
import matplotlib.pyplot as plt

# unit: m, m/s
# calibrated Macro parameters
max_corr_veh = 16
kc = 20.7 / 1000  # veh/m
# kc = 20.7 / 1000
# max_corr_veh = int(max_corr_len * kc) + 1
slope, intercept = 0.9607016401559993, -0.12087722017639413
# calibrated Micro parameters
init_disturbance = -1.4174327202700354  # m/s
# # vary intial disturbance
# init_disturbance = -np.arange(0, 5, 0.01)  # m/s
# a1, b1 = 0.0796, 0.3906
a1, b1 = 0.0866, 0.4179
lamda1 = a1 * np.log(kc) + b1
lamda3 = 0.009281678222649201

# a2, b2 = 0.142, 0.548
# results from micro calibration
# a2, b2, capacity = 0.08602678854205158, 0.32610902608664927, 1718.073982919911
a2, b2, capacity = 0.08707084260286868, 0.32955542137404753, 1731.2134133365917
lamda2 = a2 * np.log(kc) + b2
# correct the lambda3
# lamda3 = (1 - slope) * lamda1 / slope
print(f"lamda1={lamda1}, lamda3={lamda3}")
slope_est, intercept_est = lamda1 / (lamda1 + lamda3), -lamda2 / (lamda1 + lamda3) / 2
print(f"slope_est={slope_est}, intercept_est={intercept_est} and slope={slope}, intercept={intercept}")
# slope, intercept = slope_est, intercept_est
# estimate the capacity drop coefficient
term1 = (1 - np.power(slope, 2 * max_corr_veh)) * np.power(slope * init_disturbance + intercept - init_disturbance,
                                                           2) / (
                1 - np.power(slope, 2))
term3 = (slope * np.power(init_disturbance, 2) + slope * (slope * init_disturbance + intercept - init_disturbance) / (
        1 - slope)) * (1 - np.power(slope, max_corr_veh)) / (1 - slope) - slope * (
                slope * init_disturbance + intercept - init_disturbance) * max_corr_veh * pow(slope,
                                                                                              max_corr_veh) / (
                1 - slope)
cd_coef = np.exp(-lamda1 * term1 - lamda3 * term3)
# # plot the capacity drop coefficient vs. initial disturbance
# plt.plot(init_disturbance, (1 - cd_coef) * 100, c='b')
# # plot the emipirical capacity drop coefficient 17.51126126126126 %
# plt.plot(init_disturbance, np.ones_like(init_disturbance) * 17.51126126126126, c='k', lw=2)
# plt.xlabel('Initial disturbance (m/s)')
# plt.ylabel('Capacity drop coefficient (%)')
# plt.show()
print("theoretical capacity drop coefficient: ", (1 - cd_coef) * 100,
      "% and the error with the real value 17.51126126126126% is ", abs((1 - cd_coef) * 100 - 17.51126126126126), "%")
# calculte the qdf with the capacity drop coefficient and the error percentage from the empirical qdf (1465)
qdf_theoretical = capacity * cd_coef
qdf_error = (qdf_theoretical - 1465) / 1465 * 100
print("theoretical qdf: ", qdf_theoretical, "error percentage: ", abs(qdf_error), "%")
# theoretical capacity drop coefficient:  15.255217481982896 % and the error with the real value 17.51126126126126% is  2.2560437792783645 %
# theoretical qdf:  1467.1130420548352 error percentage:  0.14423495254847815 %
