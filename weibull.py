from __future__ import division
import numpy as np
import seaborn as sns
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from scipy.special import gammaln

eps = 2 ** -52

sns.set_style('ticks')
sns.set_context('talk')


def plotDataBernoulli(y, stim_strength, theta_init=[.1, 1]):
    """plotDataBernoulli(y, stim_strength, theta_init = [.1,1])

    Fit data to a Weibull function using a Bernoulli LL formulation and plot it
    ----------------------------------------------------------------------------
    y:
      array-like (1d) containing a sequence of 0s (wrong) and 1s (correct)

    stim_strength:
      array-like (1d) containing a sequence of stim_strengtherences
      corresponding to the y

    theta_init:
      list length 2 with initial estimates for alpha and beta

    """

    theta = fitDataBernoulli(y, stim_strength, theta_init)

    counter = 0
    ustim_strength = np.sort(np.unique(stim_strength))
    ustim_strength = ustim_strength[ustim_strength > 0]
    prop_correct = np.zeros(len(ustim_strength))
    for i in ustim_strength:
        prop_correct[counter] = np.mean(y[stim_strength == i])
        counter += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        ustim_strength, prop_correct, marker='o', s=50, color='k')

    xs = np.linspace(np.min(ustim_strength) * .8,
                     np.max(ustim_strength) * 1.2, 100)
    pred_y = calculateWeibullP(xs, theta[0], theta[1])
    ax.add_line(plt.Line2D(xs, pred_y, color='black'))

    ax.axvline(theta[0], color='black', linestyle='--', alpha=0.5)
    ax.axhline(.816, color='black', linestyle='--', alpha=0.5)

    ax.text(.1, .975, r'$\alpha$ = %.2f' %
            theta[0], transform=ax.transAxes)
    ax.text(.1, .925, r'$\beta$ = %.2f' %
            theta[1], transform=ax.transAxes)

    ax.set_xlim(np.min(ustim_strength) * .5, np.max(ustim_strength) * 1.1)
    ax.set_xlim(.01, 1)
    ax.set_ylim((.45, 1.05))
    ax.set_xticks([10**np.arange(-3, 0)])
    ax.set_xlabel('Stim strength')
    ax.set_ylabel('Proportion correct')

    # trim won't work with x log axis so just leave it
    sns.despine(offset=5)

    ax.set_xscale('log')

    plt.tight_layout()


def plotDataBinomial(data):
    """plotDataBinomial(data, theta_init = [.1,1])

    Fit data to a Weibull function using a Binomial LL formulation and plot it
    ---------------------------------------------------------------------------
    data:
      2d array, with n rows and 4 columns

      1st column = stim_strengths
      2nd column = proportion correct
      3rd column = number of successes
      4th column = total number of trials

    theta_init:
      list length 2 with initial estimates for alpha and beta

    """

    theta = fitDataBinomial(data)

    ustim_strength = data[:, 0]
    prop_correct = data[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        ustim_strength, prop_correct, marker='o', s=50, color='k')

    xs = np.linspace(np.min(ustim_strength) * .8,
                     np.max(ustim_strength) * 1.2, 100)
    pred_y = calculateWeibullP(xs, theta[0], theta[1])
    ax.add_line(plt.Line2D(xs, pred_y, color='black'))

    ax.axvline(theta[0], color='black', linestyle='--', alpha=0.5)
    ax.axhline(.816, color='black', linestyle='--', alpha=0.5)

    ax.text(.1, .975, r'$\alpha$ = %.2f' %
            theta[0], transform=ax.transAxes)
    ax.text(.1, .925, r'$\beta$ = %.2f' %
            theta[1], transform=ax.transAxes)

    ax.set_xlim(np.min(ustim_strength) * .5, np.max(ustim_strength) * 1.1)
    ax.set_xlim(.01, 1)
    ax.set_ylim((.45, 1.05))
    ax.set_xticks([10**np.arange(-3, 0)])
    ax.set_xlabel('Stim strength')
    ax.set_ylabel('Proportion correct')
    sns.despine(offset=5, bottom=False)

    # trim won't work with x log axis so just leave it
    sns.despine(offset=5)

    plt.tight_layout()


def fitDataBernoulli(y, stim_strength, theta_init=[.1, 1]):
    """fitDataBernoulli(y, stim_strength, theta_init = [.1,1])

    Fits data to a Weibull function using a Bernoulli LL formulation
    ----------------------------------------------------------------------------
    y:
      array-like (1d) containing a sequence of 0s (wrong) and 1s (correct)

    stim_strength:
      array-like (1d) containing a sequence of stim_strengtherences
      corresponding to the y

    theta_init:
      list length 2 with initial estimates for alpha and beta

    """
    theta = fmin(
        calculateWeibullNegLLBernoulli, theta_init,
        args=(y, stim_strength), disp=True)
    return(theta)


def fitDataBernoulliWithLapse(y, stim_strength, theta_init=[.1, 1, .05]):
    """fitDataBernoulliWithLapse(y, stim_strength, theta_init = [.1,1,.05])

    Fits data to a Weibull function with lapse using a Bernoulli LL formulation
    ----------------------------------------------------------------------------
    y:
      array-like (1d) containing a sequence of 0s (wrong) and 1s (correct)

    stim_strength:
      array-like (1d) containing a sequence of stim_strengths
      corresponding to the y

    theta_init:
      list length 3 with initial estimates for alpha, beta and lambda

    """
    theta = fmin(
        calculateWeibullNegLLBernoulliLapse, theta_init,
        args=(y, stim_strength), disp=True)
    return(theta)


def fitDataBinomial(data, theta_init=[.1, 1]):
    """fitDataBinomial(data, theta_init = [.1,1])

    Fits data to a Weibull function using a Binomial LL formulation
    ---------------------------------------------------------------------------
    data:
      2d array, with n rows and 4 columns

      1st column = stim_strengths
      2nd column = proportion correct
      3rd column = number of successes
      4th column = total number of trials

    theta_init:
      list length 2 with initial estimates for alpha and beta

    """

    theta = fmin(calculateWeibullNegLLBinomial, theta_init, args=(data,))
    return(theta)


def calculateWeibullP(stim_strength, alpha, beta):
    return(1 - .5 * np.exp(-(stim_strength / alpha) ** beta))


def calculateWeibullPLapse(stim_strength, alpha, beta, lambd):
    return(.5 + (.5 - lambd) * (1 - np.exp(-(stim_strength / alpha) ** beta)))


def calculateWeibullNegLLBernoulli(theta, y, stim_strength):
    alpha, beta = theta
    LL = np.sum(
        y * np.log(calculateWeibullP(stim_strength, alpha, beta)) +
        (1 - y) * np.log(.5 * np.exp(-(stim_strength / alpha) ** beta)))
    return(-LL)


def calculateWeibullNegLLBernoulliLapse(theta, y, stim_strength):
    alpha, beta, lambd = theta
    lambd = 1 - np.mean(y[stim_strength == .512])
    LL = np.sum(
        y * np.log(
            calculateWeibullPLapse(stim_strength, alpha, beta, lambd)) +
        (1 - y) * (np.log(
            1 - (-lambd * (1 - np.exp(-(stim_strength / alpha) ** beta))))))
    return(-LL)


def calculateWeibullNegLLBinomial(theta, data):
    alpha, beta = theta

    stim_strength = data[:, 0]
    n1_observed = data[:, 2]
    n_total = data[:, 3]

    p_pred = calculateWeibullP(stim_strength, alpha, beta)
    LL = np.sum(
        gammaln(n_total + 1) -
        gammaln(n1_observed + 1) -
        gammaln(n_total - n1_observed + 1) +
        n1_observed * np.log(p_pred + eps) +
        (n_total - n1_observed) * np.log(1 - p_pred + eps))
    return(-LL)
