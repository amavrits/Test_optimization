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
#
def exp_Risk(F, R_prior, L, cost_dict,  F_reinf=0, n_grid=1_000):
    Ps = calc_Ps(F=F, R=R_prior)
    R_post_Z1 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=(F - R_prior.kwds['loc']) / R_prior.kwds['scale'],
                          b=np.inf)
    R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + F_reinf, scale=R_prior.kwds['scale'],
                          a=-np.inf,
                          b=(F - (R_prior.kwds['loc'] + F_reinf) / R_prior.kwds['scale']))
    Pf_Z1 = calc_Pf(R_post_Z1, L, n_grid=n_grid)
    Pf_Z0 = calc_Pf(R_post_Z0, L, n_grid=n_grid)
    exp_risk = cost_dict['test'] + Ps * Pf_Z1 * cost_dict['failure'] +\
               (1 - Ps) * Pf_Z0 * (cost_dict['test_failure'] + cost_dict['failure']) +\
               (1 - Ps) * (1 - Pf_Z0) * cost_dict['test_failure']

    # expected_A = Ps * Pf_Z1 * (cost_dict['failure'] + cost_dict['test'])
    # expected_B = Ps * (1 - Pf_Z1) * (cost_dict['test'])
    # expected_C = (1 - Ps) * Pf_Z0 * (cost_dict['failure'] + cost_dict['test'] + cost_dict['test_failure'])
    # expected_D = (1 - Ps) * (1 - Pf_Z0) * (cost_dict['test'] + cost_dict['test_failure'])
    #
    # risk_updated = expected_A + expected_B + expected_C + expected_D

    return exp_risk

def cost_reinf(cost_dict, F_reinf=0):
    cost_reinforcement = F_reinf * cost_dict['reinforcement']
    # return cost_reinforcement
    return 0


def exp_Risk_comp(F, R_prior, L, cost_dict, F_reinf=0, n_grid=1_000):
    Ps = calc_Ps(F=F, R=R_prior)
    R_post_Z1 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=(F - R_prior.kwds['loc']) / R_prior.kwds['scale'],
                          b=np.inf)
    R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'] + F_reinf, scale=R_prior.kwds['scale'],
                          a=-np.inf,
                          b=(F - (R_prior.kwds['loc'] + F_reinf) / R_prior.kwds['scale']))
    Pf_Z1 = calc_Pf(R_post_Z1, L, n_grid=n_grid)
    Pf_Z0 = calc_Pf(R_post_Z0, L, n_grid=n_grid)

    risk_investment = cost_dict['test'] + (cost_dict['test_failure'] + cost_reinf(cost_dict, F_reinf=F_reinf)) * (1-Ps)
    risk_failure = cost_dict['failure'] * (Ps * Pf_Z1) + ((1 - Ps) * Pf_Z0)

    return risk_investment, risk_failure

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

R_mu = 5_000
R_sd = 500
R_prior = norm(loc=R_mu, scale=R_sd)

L_mu =4_500
L_sd = 400
L = norm(loc=L_mu, scale=L_sd)

F_reinf = 100

cost_dict = {'test': 1e3,
             'test_failure': 1e7,
             'reinforcement': 1e2,
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
for i, f in enumerate(F_grid):
    risk[i] = func(f)
    R_post_Z1 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=(f - R_prior.kwds['loc']) / R_prior.kwds['scale'],
                          b=np.inf)
    R_post_Z0 = truncnorm(loc=R_prior.kwds['loc'], scale=R_prior.kwds['scale'],
                          a=-np.inf,
                          b=(f - R_prior.kwds['loc']) / R_prior.kwds['scale'])
    Pf_Z1[i] = calc_Pf(R_post_Z1, L, n_grid=n_grid)
    Pf_Z0[i] = calc_Pf(R_post_Z0, L, n_grid=n_grid)
    Ps[i] = calc_Ps(F=f, R=R_prior)

plt.figure()
plt.plot(F_grid, risk)
plt.xlabel('${F}_{test}$')
plt.ylabel('E[Risk]')

# plt.figure()
# plt.plot(F_grid, Pf_Z1, "-", color="red")
# plt.plot(F_grid, Pf_Z0, '--', color="blue")
# plt.xlabel('${F}_{test}$')
# plt.ylabel('pfailure_posterior')
#
# plt.figure()
# plt.plot(F_grid, Ps)
# plt.xlabel('${F}_{test}$')
# plt.ylabel('psurvival')



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
