from __future__ import print_function
from __future__ import division

import numpy as np
import scipy as sp

from scipy.optimize import fmin
import matplotlib.pyplot as plt
from scipy.special import gammaln
import brewer2mpl as b2mpl
import seaborn as sns

eps = 2 ** -52

sns.set_style('ticks')
sns.set_context('talk')


def simulateData(k=15.0, A=0.67, t1_nond=0.325, t2_nond=0.375,
                 n_trials=1000, dt=0.0005):
    """
    Simulate choices and reaction times for a flat bound diffusion
    to bound model

    Parameters
    ----------
    k : float, proportionality constant between stim_strength and drift rate

    A : float, bound

    t1_nondt: float, non decision time for making t1 choice

    t2_nondt: float, non decision time for making t2 choice

    Returns
    -------
    out : 2D array of shape (N, 7) where N is number of stimulus strengths
           and the columns correspond to stim_strengths, t1_meanrt, t1_sert,
           t2_meanrt, t2_sert, n1, and n_total

    """

    stim_strengths = np.array([
        -.512, -.256, -.128, -.064, -.032, 0,
        0.032, 0.064, 0.128, 0.256, 0.512])

    max_t = 5 # this more than guarantees that all paths cross bound
    n_t = int(max_t / dt)
    t = np.arange(0, max_t, dt)

    out = np.zeros([len(stim_strengths), 7])
    for i, strength in enumerate(stim_strengths):
        print('Simulating strength {}'.format(strength))
        drift = np.ones([n_trials, n_t]) * t * k * strength
        noise = np.cumsum(np.random.randn(n_trials, n_t) * np.sqrt(dt), axis=1)
        paths = drift + noise

        bound_cross = np.abs(paths) >= A
        idx = np.apply_along_axis(lambda x: np.where(x)[0][0], 1, bound_cross)

        choice = np.sign(paths[np.arange(n_trials), idx])
        choice[choice < 0] = 0
        t1_dt = t[idx[choice == 1]]
        t2_dt = t[idx[choice == 0]]

        out[i, 0] = strength
        out[i, 1] = np.mean(t1_dt) + t1_nond
        out[i, 2] = np.std(t1_dt) / np.sqrt(len(t1_dt))
        out[i, 3] = np.mean(t2_dt) + t2_nond
        out[i, 4] = np.std(t2_dt) / np.sqrt(len(t2_dt))
        out[i, 5] = np.sum(choice)
        out[i, 6] = n_trials

    return out


def fitData(data, theta_init=[20, .5, .5, .5]):
    """
    Fits choice and reaction time data to a flat bound diffusion to bound model

    Parameters
    -------
    data : 2D array of shape (N, 7) where N is number of stimulus strengths
           and the columns correspond to stim_strengths, t1_meanrt, t1_sert,
           t2_meanrt, t2_sert, n1, and n_total

    theta_init : array, inital theta vector for fmin: k, A, t1_nond, t2_nond

    Returns
    ----------
    theta : vector of parameters: k, A, t1_nondt, t2_nondt

    """

    theta = fmin(
        calcNegLL, theta_init, args=(data,))
    return theta


def plotDataWithFit(data, theta_init=[20, .5, .5, .5]):
    """
    Fits choice and reaction time data to a flat bound diffusion to bound model
    and plots results

    Parameters
    -------
    data : 2D array of shape (N, 7) where N is number of stimulus strengths
           and the columns correspond to stim_strengths, t1_meanrt, t1_sert,
           t2_meanrt, t2_sert, n1, and n_total

    theta_init : array, inital theta vector for fmin: k, A, t1_nond, t2_nond

    Returns
    ----------
    theta : vector of parameters: k, A, t1_nondt, t2_nondt

    """

    theta = fitData(data, theta_init)

    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    xdata = np.arange(min_x, max_x, .01)
    out = calculateCPandRT(xdata, *theta)

    p_p = out['p_t1']
    t1_meanrt_p = out['t1_meanrt']
    t2_meanrt_p = out['t2_meanrt']

    p_mean = data[:, 5] / data[:, 6]
    p_se = np.sqrt(p_mean * (1 - p_mean) / data[:, 6])

    fig = plt.figure()
    colors = b2mpl.get_map('Set1', 'Qualitative', 3).mpl_colors

    # First plot choices and their DTB fits
    ax_choice = fig.add_subplot(121)
    ax_choice.add_line(plt.Line2D(xdata, p_p, color='black'))
    ax_choice.add_line(plt.Line2D(
        data[:, 0], p_mean, marker='o', color='black', linestyle='none'))
    ax_choice.errorbar(
        data[:, 0], p_mean, yerr=p_se, linestyle='None', color='black',
        capsize=0)
    ax_choice.set_xlim((np.min(xdata) * 1.2, np.max(xdata) * 1.2))
    ax_choice.set_ylim((-0.01, 1.01))
    ax_choice.set_xlabel('Stimulus strength')
    ax_choice.set_ylabel('Proportion t1 responses')

    ax_choice.text(-.6, .975, 'k = {:.2f}'.format(theta[0]), fontsize=10)
    ax_choice.text(-.6, .925, 'A = {:.2f}'.format(theta[1]), fontsize=10)
    ax_choice.text(-.6, .875, 't1_nond = {:.2f}'.format(theta[2]), fontsize=10)
    ax_choice.text(-.6, .825, 't2_nond = {:.2f}'.format(theta[3]), fontsize=10)

    # Now plot the reaction times and their DTB fits
    ax_rt = fig.add_subplot(122)
    ax_rt.add_line(plt.Line2D(xdata, t1_meanrt_p, color=colors[0]))
    ax_rt.add_line(plt.Line2D(xdata, t2_meanrt_p, color=colors[1]))
    ax_rt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], color=colors[0],
                   linestyle='None', marker='o', capsize=0)
    ax_rt.errorbar(data[:, 0], data[:, 3], yerr=data[:, 4], color=colors[1],
                   linestyle='None', marker='o', capsize=0)
    ax_rt.set_xlim(np.min(xdata) * 1.2, np.max(xdata) * 1.2)
    ax_rt.set_ylim(np.nanmin(np.concatenate([data[:, 1], data[:, 3]])) * 0.7,
                   np.nanmax(np.concatenate([data[:, 1], data[:, 3]])) * 1.1)
    ax_rt.set_xlabel('Stimulus strength')
    ax_rt.set_ylabel('Response times (s)')

    leg = ax_rt.legend(['t1 RTs', 't2 RTs'])
    for color, text in zip(colors[0:2], leg.get_texts()):
        text.set_color(color)

    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    return theta


def calculateCPandRT(stim_strengths, k, A, t1_nondt, t2_nondt):
    """
    Given a diffusion to bound model with flat bounds, compute the following
    as a function of stimulus strength:
        1) Probability of t1 choice (correct choice for positive strengths)
        2) Mean reaction times for t1 and t2 choices

    Parameters
    ----------
    stim_strengths : array of unique stimulus strengths

    k : float, proportionality constant between stim_strength and drift rate

    A : float, bound

    t1_nondt: float, non decision time for making t1 choice

    t2_nondt: float, non decision time for making t2 choice

    Returns
    -------
    out : dictionary with keys stim_strength, p_t1, t1_meanrt, t2_meanrt
    """

    # compute probability of t1 response given k and A
    p = 1 / (1 + np.exp(-2 * A * k * stim_strengths))

    # compute mean response times for t1 and t2
    t1_meanrt = np.zeros(len(stim_strengths))
    t2_meanrt = np.zeros(len(stim_strengths))
    for i in range(len(stim_strengths)):
        if stim_strengths[i] == 0:
            t1_meanrt[i] = A ** 2 + t1_nondt
            t2_meanrt[i] = A ** 2 + t2_nondt
        else:
            t1_meanrt[i] = A / (k * stim_strengths[i]) * \
                sp.tanh(A * k * stim_strengths[i]) + t1_nondt
            t2_meanrt[i] = A / (k * stim_strengths[i]) * \
                sp.tanh(A * k * stim_strengths[i]) + t2_nondt

    out = {}
    out['stim_strengths'] = stim_strengths
    out['p_t1'] = p
    out['t1_meanrt'] = t1_meanrt
    out['t2_meanrt'] = t2_meanrt
    return out


def calcNegLL(theta, data):
    """
    Given a  diffusion to bound model with flat bounds parametrized by theta,
    compute the negative log likelihood of the data, assuming binomial errors
    for choices and Gaussian errors for mean RTs

    Parameters
    ----------
    theta : 4 element list containing the DTB parameters
            k, A, t1_nondt, t2_nondt

    data : 2D array of shape (N, 7) where N is number of stimulus strengths
           and the columns correspond to stim_strengths, t1_meanrt, t1_sert,
           t2_meanrt, t2_sert, n1, and n_total

    Returns
    -------
    neg_ll : float, negative log lielihood of the data
    """

    k, A, t1_nondt, t2_nondt = theta
    N = data.shape[0]

    # observed data
    stim_strengths = data[:, 0]
    t1_meanrt_o = data[:, 1]
    t1_sert_o = data[:, 2]
    t2_meanrt_o = data[:, 3]
    t2_sert_o = data[:, 4]
    n1_o = data[:, 5]
    n_total = data[:, 6]

    # predicted data
    d = calculateCPandRT(stim_strengths, k, A, t1_nondt, t2_nondt)
    p_p, t1_meanrt_p, t2_meanrt_p = d['p_t1'], d['t1_meanrt'], d['t2_meanrt']

    # eliminate nans
    t1_idx = np.logical_not(np.isnan(t1_meanrt_o))
    t2_idx = np.logical_not(np.isnan(t2_meanrt_o))

    neg_LL = 0

    # RT cost (log of Gaussian, summed across stim_strengths)
    neg_LL += np.sum(
        (t1_meanrt_o[t1_idx] - t1_meanrt_p[t1_idx]) ** 2 /
        (2 * t1_sert_o[t1_idx] ** 2)
    ) + N / 2 * np.log(2 * np.pi) + np.sum(np.log(t1_sert_o[t1_idx]))
    neg_LL += np.sum(
        (t2_meanrt_o[t2_idx] - t2_meanrt_p[t2_idx]) ** 2 /
        (2 * t2_sert_o[t2_idx] ** 2)
    ) + N / 2 * np.log(2 * np.pi) + np.sum(np.log(t2_sert_o[t2_idx]))

    # CP cost (log of binomial, summed across stim_strengths)
    neg_LL -= np.sum(
        gammaln(n_total + 1) - gammaln(n1_o + 1) - gammaln(n_total - n1_o + 1) +
        n1_o * np.log(p_p * eps) + (n_total - n1_o) * np.log(1 - p_p + eps))

    return(neg_LL)


if __name__ == '__main__':
    theta = [12.5, 0.67, 0.325, 0.375]
    data = simulateData(*theta)
    fitted_theta = fitData(data)
    print('---' * 20)
    print('---' * 20)
    print('True k: {:.2f}, Fitted k: {:.2f}'.format(theta[0], fitted_theta[0]))
    print('True A: {:.2f}, Fitted A: {:.2f}'.format(theta[1], fitted_theta[1]))
    print('True t1_nond: {:.2f}, Fitted t1_nond: {:.2f}'.format(
        theta[2], fitted_theta[2]))
    print('True t2_nond: {:.2f}, Fitted t2_nond: {:.2f}'.format(
        theta[3], fitted_theta[3]))
    print('---' * 20)
    print('---' * 20)
