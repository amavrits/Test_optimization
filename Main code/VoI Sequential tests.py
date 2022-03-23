import numpy as np
from scipy.optimize import minimize
from scipy.integrate import trapz
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def calc_Ps(F, R):
    return 1-R.cdf(F)

def calc_Pf(R, L, n_grid=1_000):
    x_grid = np.linspace(0, 2*R.kwds['loc'], n_grid)
    return trapz(R.cdf(x_grid) * L.pdf(x_grid), x_grid)

def cost_reinf(cost_dict, R_reinf=0):
    cost_reinforcement = R_reinf * cost_dict['reinforcement']
    return cost_reinforcement

def exp_Risk(F, R_prior, L, cost_dict, R_reinf=0, n_grid=1_000):
    Ps = calc_Ps(F=F, R=R_prior)
    R_post_Z1 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=(F - R_prior.kwds['loc']) / R_prior.kwds['scale'], b=np.inf)
    # R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + R_reinf, scale=R_prior.kwds['scale'],
    #                       a=-np.inf, b=(F - (R_prior.kwds['loc'] + R_reinf)) / R_prior.kwds['scale'])
    # R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + R_reinf, scale=R_prior.kwds['scale'], a=-np.inf, b=+np.inf)
    R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + R_reinf, scale=R_prior.kwds['scale'],
                          a=(F-R_prior.kwds['loc'])/R_prior.kwds['scale'], b=+np.inf)
    Pf_Z1 = calc_Pf(R_post_Z1, L, n_grid=n_grid)
    Pf_Z0 = calc_Pf(R_post_Z0, L, n_grid=n_grid)
    exp_risk = cost_dict['test'] + Ps * Pf_Z1 * cost_dict['failure'] +\
               (1 - Ps) * Pf_Z0 * (cost_dict['test_failure'] + cost_dict['failure'] + cost_reinf(cost_dict=cost_dict, R_reinf=R_reinf)) +\
               (1 - Ps) * (1 - Pf_Z0) * (cost_dict['test_failure'] + cost_reinf(cost_dict=cost_dict, R_reinf=R_reinf))

    # expected_A = Ps * Pf_Z1 * (cost_dict['failure'] + cost_dict['test'])
    # expected_B = Ps * (1 - Pf_Z1) * (cost_dict['test'])
    # expected_C = (1 - Ps) * Pf_Z0 * (cost_dict['failure'] + cost_dict['test'] + cost_dict['test_failure'])
    # expected_D = (1 - Ps) * (1 - Pf_Z0) * (cost_dict['test'] + cost_dict['test_failure'])
    # exp_risk = expected_A + expected_B + expected_C + expected_D

    return exp_risk

def exp_Risk_comp(F, R_prior, L, cost_dict, R_reinf=0, n_grid=1_000):
    Ps = calc_Ps(F=F, R=R_prior)
    R_post_Z1 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=(F - R_prior.kwds['loc']) / R_prior.kwds['scale'], b=np.inf)
    # R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + R_reinf, scale=R_prior.kwds['scale'],
    #                       a=-np.inf, b=(F - (R_prior.kwds['loc'] + R_reinf)) / R_prior.kwds['scale'])
    # R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + R_reinf, scale=R_prior.kwds['scale'], a=-np.inf, b=+np.inf)
    R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + R_reinf, scale=R_prior.kwds['scale'],
                          a=(F-R_prior.kwds['loc'])/R_prior.kwds['scale'], b=+np.inf)
    Pf_Z1 = calc_Pf(R_post_Z1, L, n_grid=n_grid)
    Pf_Z0 = calc_Pf(R_post_Z0, L, n_grid=n_grid)

    risk_investment = cost_dict['test'] + (cost_dict['test_failure'] + cost_reinf(cost_dict=cost_dict, R_reinf=R_reinf)) * (1-Ps)
    risk_failure = cost_dict['failure'] * ((Ps * Pf_Z1) + ((1 - Ps) * Pf_Z0))

    return risk_investment, risk_failure

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

R_mu = 6_000
R_sd = 1_000
R_prior = norm(loc=R_mu, scale=R_sd)

L_mu = 2_700
L_sd = 400
L = norm(loc=L_mu, scale=L_sd)

R_reinf = 500

cost_dict = {'test': 1e2,
             'test_failure': 1e3,
             'reinforcement': 1e3,
             'failure': 1e8}

n_grid = 100

Pf_prior = calc_Pf(R=R_prior, L=L, n_grid=n_grid)
Risk_prior = Pf_prior * cost_dict['failure']

n_test_rounds = 2
F_opt = np.zeros(n_test_rounds)
R_Z1 = R_prior
R_Z0 = R_prior

def sequential_exp_Risk(F_test):

    exp_Risk_1 = exp_Risk(F=F_test[0], R_prior=R_prior, L=L, cost_dict=cost_dict, R_reinf=R_reinf, n_grid=n_grid)

    Ps = calc_Ps(F=F_test[0], R=R_prior)
    R_post_Z1 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=(F_test[0] - R_prior.kwds['loc']) / R_prior.kwds['scale'], b=np.inf)
    # R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + R_reinf, scale=R_prior.kwds['scale'], a=-np.inf, b=+np.inf)
    R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + R_reinf, scale=R_prior.kwds['scale'],
                          a=(F_test[0]-R_prior.kwds['loc'])/R_prior.kwds['scale'], b=+np.inf)

    exp_Risk_2Z1 = exp_Risk(F=F_test[1], R_prior=R_post_Z1, L=L, cost_dict=cost_dict, R_reinf=R_reinf, n_grid=n_grid)
    exp_Risk_2Z0 = exp_Risk(F=F_test[1], R_prior=R_post_Z0, L=L, cost_dict=cost_dict, R_reinf=R_reinf, n_grid=n_grid)
    exp_Risk_2 = Ps * exp_Risk_2Z1 + (1 - Ps) * exp_Risk_2Z0

    return exp_Risk_1 + exp_Risk_2

optimizer = minimize(sequential_exp_Risk, x0=np.array([5_000, 5_000]))
F_opt = optimizer['x']

n_mesh = 100
F1_mesh, F2_mesh = np.meshgrid(np.linspace(0, 10_000, n_mesh), np.linspace(0, 10_000, n_mesh+1))

exp_risk = np.zeros(F1_mesh.shape[0] * F1_mesh.shape[1])
for i, (f1, f2) in enumerate(zip(F1_mesh.flatten(), F2_mesh.flatten())):
    exp_risk[i] = sequential_exp_Risk(np.stack((f1, f2)))


fig = plt.figure()
plt.contourf(F1_mesh, F2_mesh, exp_risk.reshape(F1_mesh.shape[0], F1_mesh.shape[1]))
plt.axvline(F_opt[0], color='r')
plt.axhline(F_opt[1], color='r')
plt.title('Expected total risk', fontsize=16)
plt.xlabel('${F}_{test,1}$ [kN]', fontsize=16)
plt.ylabel('${F}_{test,2}$ [kN]', fontsize=16)


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(F1_mesh, F2_mesh, exp_risk.reshape(F1_mesh.shape[0], F1_mesh.shape[1]),
                rstride=1, cstride=1, cmap='viridis', edgecolor='none')
