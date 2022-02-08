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

def cost_reinf(cost_dict, F_reinf=0):
    cost_reinforcement = F_reinf * cost_dict['reinforcement']
    return cost_reinforcement

def exp_Risk(F, R_prior, L, cost_dict, F_reinf=0, n_grid=1_000):
    Ps = calc_Ps(F=F, R=R_prior)
    R_post_Z1 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=(F - R_prior.kwds['loc']) / R_prior.kwds['scale'], b=np.inf)
    # R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + F_reinf, scale=R_prior.kwds['scale'],
    #                       a=-np.inf, b=(F - (R_prior.kwds['loc'] + F_reinf)) / R_prior.kwds['scale'])
    R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + F_reinf, scale=R_prior.kwds['scale'], a=-np.inf, b=+np.inf)
    Pf_Z1 = calc_Pf(R_post_Z1, L, n_grid=n_grid)
    Pf_Z0 = calc_Pf(R_post_Z0, L, n_grid=n_grid)
    exp_risk = cost_dict['test'] + Ps * Pf_Z1 * cost_dict['failure'] +\
               (1 - Ps) * Pf_Z0 * (cost_dict['test_failure'] + cost_dict['failure'] + cost_reinf(cost_dict=cost_dict, F_reinf=F_reinf)) +\
               (1 - Ps) * (1 - Pf_Z0) * (cost_dict['test_failure'] + cost_reinf(cost_dict=cost_dict, F_reinf=F_reinf))

    # expected_A = Ps * Pf_Z1 * (cost_dict['failure'] + cost_dict['test'])
    # expected_B = Ps * (1 - Pf_Z1) * (cost_dict['test'])
    # expected_C = (1 - Ps) * Pf_Z0 * (cost_dict['failure'] + cost_dict['test'] + cost_dict['test_failure'])
    # expected_D = (1 - Ps) * (1 - Pf_Z0) * (cost_dict['test'] + cost_dict['test_failure'])
    # exp_risk = expected_A + expected_B + expected_C + expected_D

    return exp_risk

def exp_Risk_comp(F, R_prior, L, cost_dict, F_reinf=0, n_grid=1_000):
    Ps = calc_Ps(F=F, R=R_prior)
    R_post_Z1 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=(F - R_prior.kwds['loc']) / R_prior.kwds['scale'], b=np.inf)
    # R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + F_reinf, scale=R_prior.kwds['scale'],
    #                       a=-np.inf, b=(F - (R_prior.kwds['loc'] + F_reinf)) / R_prior.kwds['scale'])
    R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + F_reinf, scale=R_prior.kwds['scale'], a=-np.inf, b=+np.inf)
    Pf_Z1 = calc_Pf(R_post_Z1, L, n_grid=n_grid)
    Pf_Z0 = calc_Pf(R_post_Z0, L, n_grid=n_grid)

    risk_investment = cost_dict['test'] + (cost_dict['test_failure'] + cost_reinf(cost_dict=cost_dict, F_reinf=F_reinf)) * (1-Ps)
    risk_failure = cost_dict['failure'] * ((Ps * Pf_Z1) + ((1 - Ps) * Pf_Z0))

    return risk_investment, risk_failure

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

R_mu = 5_000
R_sd = 500
R_prior = norm(loc=R_mu, scale=R_sd)

L_mu =4_500
L_sd = 400
L = norm(loc=L_mu, scale=L_sd)

F_reinf = 500

cost_dict = {'test': 1e3,
             'test_failure': 1e6,
             'reinforcement': 1e4,
             'failure': 1e8}

n_grid = 100

Pf_prior = calc_Pf(R=R_prior, L=L, n_grid=n_grid)
Risk_prior = Pf_prior * cost_dict['failure']

func = lambda F: exp_Risk(F=F, R_prior=R_prior, L=L, cost_dict=cost_dict, F_reinf=F_reinf, n_grid=n_grid)

# optimizer = minimize(func, x0=5000)
# F_opt = optimizer['x']

F_grid = np.linspace(0, 10_000, 100)
risk = np.zeros_like(F_grid)
Pf_Z1 = np.zeros_like(F_grid)
Pf_Z0 = np.zeros_like(F_grid)
Ps = np.zeros_like(F_grid)
x = np.linspace(0, 10_000, 100)
for i, f in enumerate(F_grid):
    risk[i] = func(f)

plt.figure()
plt.plot(F_grid, risk)
plt.xlabel('${F}_{test}$')
plt.ylabel('E[Risk]')

F_grid = np.linspace(0, 10_000, 100)
inv_risk = np.zeros_like(F_grid)
fail_risk = np.zeros_like(F_grid)
for i, f in enumerate(F_grid):
    inv_risk[i], fail_risk[i] = exp_Risk_comp(F=f, R_prior=R_prior, L=L, cost_dict=cost_dict, F_reinf=F_reinf, n_grid=n_grid)

plt.figure()
plt.plot(F_grid, inv_risk, label='Investment risk')
plt.plot(F_grid, fail_risk, label='Failure risk')
plt.plot(F_grid, inv_risk+fail_risk, label='Total risk')
plt.xlabel('${F}_{test}$')
plt.ylabel('E[Risk]')
plt.legend()



# func = lambda F_vec: exp_Risk(F=F_vec[0], F_reinf=F_vec[1], cost_dict=cost_dict, R_prior=R_prior, L=L, n_grid=n_grid)
#
# # optimizer = minimize(func, x0=np.array([5000,1000]))
# # F_vec_opt = optimizer['x']
#
# n = 50
# F_test_grid = np.linspace(0, 10_000, n)
# F_reinf_grid = np.linspace(0, 5_000, n+1)
# F_test_mesh, F_reinf_mesh = np.meshgrid(F_test_grid, F_reinf_grid)
#
# risk_inv = np.zeros(F_test_mesh.shape[0]*F_test_mesh.shape[1])
# risk_fail = np.zeros(F_test_mesh.shape[0]*F_test_mesh.shape[1])
# exp_risk = np.zeros(F_test_mesh.shape[0]*F_test_mesh.shape[1])
# for i, (f_test, f_reinf) in enumerate(zip(F_test_mesh.flatten(), F_reinf_mesh.flatten())):
#     exp_risk[i] = func(np.stack((f_test, f_reinf)))
#     risk_inv[i], risk_fail[i] = exp_Risk_comp(F=f_test, R_prior=R_prior, L=L, cost_dict=cost_dict, F_reinf=f_reinf, n_grid=100)
#
# exp_risk2 = risk_inv + risk_fail
#
# fig = plt.figure()
# plt.contourf(F_test_mesh, F_reinf_mesh, exp_risk.reshape(F_test_mesh.shape[0], F_test_mesh.shape[1]))
# plt.xlabel('${F}_{test}$', fontsize=16)
# plt.ylabel('${F}_{reinf}$', fontsize=16)
#
# fig, ax = plt.subplots(1, 3)
# ax[0].contourf(F_test_mesh, F_reinf_mesh, risk_inv.reshape(F_test_mesh.shape[0], F_test_mesh.shape[1]))
# ax[0].set_xlabel('${F}_{test}$', fontsize=16)
# ax[0].set_ylabel('${F}_{reinf}$', fontsize=16)
# ax[0].set_title('Investment risk', fontsize=16)
# ax[1].contourf(F_test_mesh, F_reinf_mesh, risk_fail.reshape(F_test_mesh.shape[0], F_test_mesh.shape[1]))
# ax[1].set_xlabel('${F}_{test}$', fontsize=16)
# ax[1].set_ylabel('${F}_{reinf}$', fontsize=16)
# ax[1].set_title('Failure risk', fontsize=16)
# ax[2].contourf(F_test_mesh, F_reinf_mesh, exp_risk2.reshape(F_test_mesh.shape[0], F_test_mesh.shape[1]))
# ax[2].set_xlabel('${F}_{test}$', fontsize=16)
# ax[2].set_ylabel('${F}_{reinf}$', fontsize=16)
# ax[2].set_title('Total risk', fontsize=16)
