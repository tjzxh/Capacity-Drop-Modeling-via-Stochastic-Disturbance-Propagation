# Be careful with the unit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute, basinhopping
from scipy.special import erf
from numpy import sqrt, pi, exp, log
from scipy.optimize import Bounds
import polars as pl


# define the log-likelihood function
def loglikelihood(x):
    ll1, ll2, ll3 = x[0], x[1], x[2]
    l1, l2, l3 = ll1 / v1, ll2 / v2, ll3 / v3
    va = 0  # test the necessary of va
    H = l1 * (vs - vl) ** 2 + l2 * vs + l3 * (vs - va) ** 2
    A = (-2 * l1 * vl + l2 - 2 * l3 * va) / (2 * (l1 + l3))
    B = -A ** 2 + (l1 * vl ** 2 + l3 * va ** 2) / (l1 + l3)
    Z = (sqrt(pi) / 2) * exp(-(l1 + l3) * B) / sqrt(l1 + l3) * (erf(sqrt(l1 + l3) * (vm + A)) - erf(sqrt(l1 + l3) * A))
    logZ = log(Z)
    min_obj = np.sum(H) + np.sum(logZ)
    return min_obj


# read the homo data, unit: ft/s
# 'frame', 'subjectSpeed', 'leaderSpeed', 'speedDiff', 'spacing', 'subjectClass', 'relativeClass', 'Vehlength', 'meanSpeed'
pl_data = pl.read_parquet("../data/I24_lane2_homo_data.parquet")
df = pl_data.select(['frame', 'subjectSpeed', 'leaderSpeed', 'speedDiff', 'meanSpeed', 'net_spacing', 'subjectClass',
                     'relativeClass'])
# convert ft to m
ft_to_m = 0.3048
# alter the unit m/s (1) OR km/h (2)
unit_flag = 2
ms_to_kmh, m_to_km = 1, 1
if unit_flag == 2:
    ms_to_kmh, m_to_km = 3.6, 1 / 1000
df = df.with_columns(v=pl.col('subjectSpeed') * ft_to_m * ms_to_kmh, vl=pl.col('leaderSpeed') * ft_to_m * ms_to_kmh,
                     vd=pl.col('speedDiff') * ft_to_m * ms_to_kmh, va=pl.col('meanSpeed') * ft_to_m * ms_to_kmh,
                     s=pl.col('net_spacing') * ft_to_m)
# [frame, subject speed, leader speed, speed difference, average speed in lane, spacing, subject class, relative class]
# train_data = data[:int(data.shape[0] * 2 / 3)]
vm = ms_to_kmh * 120 / 3.6  # max speed (m/s)
result = []
# calculate the normalized parameters
v1, v2, v3 = np.percentile(df['vd'] ** 2, 99), np.percentile(df['v'], 99), np.percentile((df['v'] - df['va']) ** 2,
                                                                                         99)
print(v1, v2, v3)
# iterate similar spacing (units:m)
# # plot the spacing distribution
# plt.hist(df['s'], bins=100)
# plt.show()
space_interval = 0.5  # m
for i in np.arange(0, df['s'].max(), space_interval):
    df_spacing = df.filter((df['s'] > i) & (df['s'] <= i + space_interval))
    if df_spacing.shape[0] < 200:
        continue

    vs, vl, va = df_spacing['v'].to_numpy(), df_spacing['vl'].to_numpy(), df_spacing['va'].to_numpy()
    # minimize the objective function
    x0 = np.array([0.1, -0.001, 0.01])
    bnds = Bounds([0, -np.inf, 0], [np.inf, np.inf, np.inf])
    res = minimize(loglikelihood, x0, options={'disp': True}, bounds=bnds, method='Nelder-Mead', tol=1e-6)
    # # try the global optimization
    # bounds = [(0, 10), (-10, 10), (0, 10), (0, 200 / 3.6)]
    # res = optimize.brute(loglikelihood, bounds, full_output=True, finish=optimize.fmin)
    result.append([m_to_km * df_spacing['s'].mean(), len(vs), res.x[0] / v1, res.x[1] / v2, res.x[2] / v3, res.fun])
file_name = '../data/I24_lamda_lane2_m.txt' if unit_flag == 1 else '../data/I24_lamda_lane2_km.txt'
np.savetxt(file_name, np.array(result))
