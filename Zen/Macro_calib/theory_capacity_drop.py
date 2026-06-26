import numpy as np
import matplotlib.pyplot as plt

# unit: m, m/s
# calibrated Macro parameters
max_corr_veh = 12
kc = 28.2 / 1000  # veh/m
max_corr_len = 428.5512539184953
max_corr_veh_est = int(max_corr_len * kc) + 1
print("The estimated max_corr_veh is", max_corr_veh_est)
slope, intercept = 0.8226758861911331, 0.6037869445793889
# calibrated Micro parameters
init_disturbance = -0.7580311736263425  # m/s
# # vary intial disturbance
# init_disturbance = -np.arange(0, 5, 0.01)  # m/s
a1, b1 = 0.1716, 0.7681
lamda1 = a1 * np.log(kc) + b1
lamda3 = 0.051074896197252764

# a2, b2 = 0.142, 0.548
# results from micro calibration
a2, b2, capacity = 0.06663314612341437, 0.21759390582317092, 1917.2139688101033
lamda2 = a2 * np.log(kc) + b2
# correct the lambda3
# lamda3 = (1 - slope) * lamda1 / slope
print(f"lamda1={lamda1}, lamda3={lamda3}")
slope_est, intercept_est = lamda1 / (lamda1 + lamda3), -lamda2 / (lamda1 + lamda3) / 2
print(f"slope_est={slope_est}, intercept_est={intercept_est} and slope={slope}, intercept={intercept}")
slope, intercept = slope_est, intercept_est
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
      "% and the error with the real value 21.499999999999996% is ", abs((1 - cd_coef) * 100 - 21.499999999999996), "%")
# theoretical capacity drop coefficient:  20.827108000742477 %
# calculte the qdf with the capacity drop coefficient and the error percentage from the empirical qdf (1507.2)
qdf_theoretical = capacity * cd_coef
qdf_error = (qdf_theoretical - 1507.2) / 1507.2 * 100
print("theoretical qdf: ", qdf_theoretical, "error percentage: ", abs(qdf_error), "%")

# slope_est=0.7530606483908342, intercept_est=0.04568888978866672 and slope=0.8226758861911331, intercept=0.6037869445793889
# theoretical capacity drop coefficient:  20.827108000742477 % and the error with the real value 21.499999999999996% is  3.3158467394812163 %
# theoretical qdf:  1508.4933711588815 error percentage:  0.08581284228247264 %
