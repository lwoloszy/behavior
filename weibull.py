from __future__ import division
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.special import gammaln

eps = 2 ** -52


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


def plotWeibullBernoulli(y, stim_strength, theta_init=[.1, 1]):
    theta = fitDataBernoulli(y, stim_strength, theta_init)

    counter = 0
    stim_strengtherences = np.sort(np.unique(stim_strength))
    propCorrect = np.zeros(len(stim_strengtherences))
    for i in stim_strengtherences:
        propCorrect[counter] = np.mean(y[stim_strength == i])
        counter += 1

    fig = plt.figure()
    nullfmt = NullFormatter()

    left, width = 0.15, 0.75
    bottom, height = 0.1, 0.85
    rect1 = [left, bottom, width, height]
    rect2 = [left - 0.05, bottom, 0.04, height]

    axMain = fig.add_axes(rect1)
    axMain.scatter(
        stim_strengtherences[0:], propCorrect[0:], marker='o', s=50, color='k')

    xs = np.arange(0, 1, .001)
    fittedYs = calculateWeibullP(xs, theta[0], theta[1])
    axMain.add_line(plt.Line2D(xs, fittedYs, linewidth=2, color='black'))

    axMain.axvline(theta[0], color='black', linestyle='--')
    axMain.axhline(.816, color='black', linestyle='--')

    axMain.set_xlim((0.02, 1))
    axMain.set_ylim((.35, 1.05))
    axMain.set_title('Performance as a Function of Unsigned Stim_Strengtherence')
    axMain.set_xlabel('Unsigned Stim_Strengtherence (%)')
    axMain.set_xscale('log')
    axMain.yaxis.set_major_formatter(nullfmt)

    axMain.text(.1, .95, 'alpha = %.2f' %
                theta[0], transform=axMain.transAxes)
    axMain.text(.1, .925, 'beta = %.2f' %
                theta[1], transform=axMain.transAxes)

    # For zero percent stim_strengtherence (can't be on log axis)
    axSide = fig.add_axes(rect2)
    axSide.scatter(0, 0.5, marker='o', s=50, color='black')

    xs = np.arange(0, 1, .001)
    fittedYs = calculateWeibullP(xs, theta[0], theta[1])
    axSide.add_line(plt.Line2D(xs, fittedYs, linewidth=2, color='black'))

    axSide.set_ylabel('Proportion Correct')
    axSide.set_xlim((-.0001, .001))
    axSide.set_ylim((.35, 1.05))
    axSide.xaxis.set_major_formatter(nullfmt)


def plotWeibullBernoulliWithLapse(y, stim_strength, theta_init=[.1, 1, .05]):
    theta = fitDataBernoulliWithLapse(y, stim_strength, theta_init)
    print theta

    counter = 0
    stim_strengtherences = np.sort(np.unique(stim_strength))
    propCorrect = np.zeros(len(stim_strengtherences))
    for i in stim_strengtherences:
        propCorrect[counter] = np.mean(y[stim_strength == i])
        counter += 1

    fig = plt.figure()
    nullfmt = NullFormatter()

    left, width = 0.15, 0.75
    bottom, height = 0.1, 0.85
    rect1 = [left, bottom, width, height]
    rect2 = [left - 0.05, bottom, 0.04, height]

    axMain = fig.add_axes(rect1)
    axMain.scatter(
        stim_strengtherences[0:], propCorrect[0:], marker='o', s=50, color='k')

    xs = np.arange(0, 1, .001)
    fittedYs = calculateWeibullPLapse(
        xs, theta[0], theta[1], theta[2])
    axMain.add_line(plt.Line2D(xs, fittedYs, linewidth=2, color='black'))

    axMain.axvline(theta[0], color='black', linestyle='--')
    axMain.axhline(.816, color='black', linestyle='--')

    axMain.set_xlim((0.02, 1))
    axMain.set_ylim((.35, 1.05))
    axMain.set_title('Performance as a Function of Unsigned Stim_Strengtherence')
    axMain.set_xlabel('Unsigned Stim_Strengtherence (%)')
    axMain.set_xscale('log')
    axMain.yaxis.set_major_formatter(nullfmt)

    axMain.text(.1, .95, 'alpha = %.2f' %
                theta[0], transform=axMain.transAxes)
    axMain.text(.1, .925, 'beta = %.2f' %
                theta[1], transform=axMain.transAxes)
    axMain.text(.1, .900, 'lambda = %.2f' %
                theta[2], transform=axMain.transAxes)

    # For zero percent stim_strengtherence (can't be on log axis)
    axSide = fig.add_axes(rect2)
    axSide.scatter(0, 0.5, marker='o', s=50, color='black')

    xs = np.arange(0, 1, .001)
    fittedYs = calculateWeibullP(xs, theta[0], theta[1])
    axSide.add_line(plt.Line2D(xs, fittedYs, linewidth=2, color='black'))

    axSide.set_ylabel('Proportion Correct')
    axSide.set_xlim((-.0001, .001))
    axSide.set_ylim((.35, 1.05))
    axSide.xaxis.set_major_formatter(nullfmt)


def plotWeibullBinomial(data):
    theta = fitDataBinomial(data)

    stim_strengtherences = data[:, 0]
    propCorrect = data[:, 1]

    fig = plt.figure()
    nullfmt = NullFormatter()

    left, width = 0.15, 0.75
    bottom, height = 0.1, 0.85
    rect1 = [left, bottom, width, height]
    rect2 = [left - 0.05, bottom, 0.04, height]

    axMain = fig.add_axes(rect1)
    axMain.scatter(stim_strengtherences, propCorrect, marker='o', s=50)

    xs = np.arange(.01, 1, .001)
    fittedYs = calculateWeibullP(xs, theta[0], theta[1])
    axMain.add_line(plt.Line2D(xs, fittedYs, linewidth=2))

    axMain.axvline(theta[0], color='black', linestyle='--')
    axMain.axhline(.816, color='black', linestyle='--')

    axMain.set_xlim((.1, 1))
    axMain.set_ylim((.35, 1.05))
    axMain.set_title('Performance as a Function of Unsigned Stim_Strengtherence')
    axMain.set_xlabel('Unsigned Stim_Strengtherence (%)')
    axMain.set_xscale('log')
    axMain.yaxis.set_major_formatter(nullfmt)

    axMain.text(1, 1, 'alpha = %.2f' % theta[0])
    axMain.text(1, .975, 'beta = %.2f' % theta[1])

    # For zero percent stim_strengtherence (can't be on log axis)
    axSide = fig.add_axes(rect2)
    axSide.scatter(0, .5, marker='o', s=50)

    xs = np.arange(0, .001, .0001)
    fittedYs = calculateWeibullP(xs, theta[0], theta[1])
    axSide.add_line(plt.Line2D(xs, fittedYs, linewidth=2))

    axSide.set_ylabel('Proportion Correct')
    axSide.set_xlim((-.0001, .001))
    axSide.set_ylim((.35, 1.05))
    axSide.xaxis.set_major_formatter(nullfmt)
