# Code to simulate and fit diffusion to bound models whose
# bounds decrease quadratically;
#
# Original version written for Matlab by Daniel Wolpert;
# Translated into Python and modified further by Luke Woloszyn

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import norm
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import seaborn as sns
import brewer2mpl as b2mpl

import fminbnd

eps = 2 ** -52

# a few global variables that speed up optimization
T = None
It = None
STIM_STRENGTH = None
Ic = None

sns.set_style('ticks')
sns.set_context('talk')


def simulateData(k=12.5, B0=0.6, Bdel=0.3, B2=0.3,
                 tnd=0.4, tnd_sd=0.05, stim_strength_bias=0, y0=0,
                 n_trials=5000, dt=0.0005):
    """
    Simulate choices and reaction times for a diffusion to bound model
    with quadratically and symmetrically decreasing bounds

    Parameters
    ----------
    k : float, proportionality constant between stim_strength and drift rate

    B0 : float, initial bound height

    Bdelay : float, delay in seconds before bound starts decreasing

    B2 : float, coefficient of quadratic term that governs bound decrease

    tnd: float, non decision time

    tnd_sd: float, standard deviation of non-decision time

    y0 : float, initial offset of decision variable (relative to 0)

    stim_strength_bias : float, bias in drift rate (added to stim_strength before k mult)

    n_trials : int, number of trials per stim_strength to simulate

    dt : temporal resolution of the simulated diffusion process

    Returns
    -------
    out : 2D array of shape (N, 3) where N is number of trials
          and the columns correspond to stim_strength, rt, and choice

    """

    stim_strengths = np.array([
        -.512, -.256, -.128, -.064, -.032, 0,
        0.032, 0.064, 0.128, 0.256, 0.512])

    max_t = 3.0 # guarantees that all paths cross bound
    n_t = int(max_t / dt)
    t = np.arange(0, max_t, dt)

    # bounds
    Bup = B0 * (t == t)
    s = t > Bdel
    Bup[s] = B0 - B2 * np.power(t[s] - Bdel, 2)
    Blo = -Bup

    out = np.zeros([len(stim_strengths) * n_trials, 3])
    counter = 0
    for i, stim_strength in enumerate(stim_strengths):
        print('Simulating strength {:.3f}'.format(stim_strength))
        drift = k * (stim_strength + stim_strength_bias)
        drift = np.ones([n_trials, n_t]) * t * drift
        noise = np.cumsum(np.random.randn(n_trials, n_t) * np.sqrt(dt), axis=1)
        paths = drift + noise
        bound_cross = np.logical_or(paths >= Bup, paths <= Blo)
        idx = np.apply_along_axis(lambda x: np.where(x)[0][0], 1, bound_cross)

        choice = np.sign(paths[np.arange(n_trials), idx])
        choice[choice < 0] = 0
        t_decision = t[idx]
        t_rt = t_decision + tnd + np.random.randn(n_trials) * tnd_sd

        out[counter:counter + n_trials, 0] = np.repeat(stim_strength, n_trials)
        out[counter:counter + n_trials, 1] = t_rt
        out[counter:counter + n_trials, 2] = choice
        counter += n_trials

    return out


def fitData(
        data, dt=.0005, k=10, B0=0.8, Bdel=.5, B2=0.1,
        tnd=0.3, tnd_sd=0.02, stim_strength_bias=0.1, y0=0.2):

    """
    Fit choices and reaction times to a diffusion to bound model
    with quadratically and symmetrically decreasing bounds

    Parameters
    ----------
    data : 2D array of shape (N, 3) where N is number of trials
          and the columns correspond to stim_strength, rt, and choice

    dt : temporal resolution of the simulated diffusion process

    ########## The following are initial values of parameters ##############

    k : float, proportionality constant between stim_strength and drift rate

    B0 : float, initial bound height

    Bdelay : float, delay in seconds before bound starts decreasing

    B2 : float, coefficient of quadratic term that governs bound decrease

    tnd: float, non decision time

    tnd_sd: float, standard deviation of non-decision time

    stim_strength_bias : bias in drift rate (added to stim_strength before k mult)

    y0 : initial offset of decision variable (relative to 0)


    Returns
    -------
    theta : 1D array of 8 params -
            k, B0, Bdel, B2, tnd, tnd_sd, stim_strength_bias, y0

    """

    global T, It, STIM_STRENGTH, Ic
    T, It, STIM_STRENGTH, Ic = None, None, None, None

    stim_strengths = data[:, 0]
    rt = data[:, 1]
    ci = data[:, 2]

    # plausible bounds for the parameters
    theta_lo = [0, 0, 0, 0, .1, .01, -np.inf, -np.inf]
    theta_hi = [np.inf, np.inf, np.max(rt), np.inf,
                np.inf, 0.5, np.inf, np.inf]
    theta_init = [k, B0, Bdel, B2, tnd, tnd_sd, stim_strength_bias, y0]

    theta, fopt = fminbnd.fminbnd(
        calculateNegLL, theta_init, args=(stim_strengths, rt, ci, dt),
        LB=theta_lo, UB=theta_hi)

    # clear the globals
    T, It, STIM_STRENGTH, Ic = None, None, None, None

    return theta


def plotDataWithFit(data, theta, dt=.0005):
    """
    Given a  diffusion to bound model with quadratically collapsing bounds
    parametrized by theta, plot raw behavioral data and its dtb fit

    Parameters
    ----------
    data : 2D array of shape (N, 3) where N is total number of trials
           and the columns correspond to stim_strength, rt, and choice

    theta : 8 element list or array containing the fitted DTB parameters
            k, B0, Bdel, B2, tnd, tnd_sd, stim_strength_bias, y0

    dt : float, temporal resolution, in seconds, to use when propagating
         the stimulus strengths to visualize (for plotting only)

    Returns
    -------

    """

    colors = b2mpl.get_map('Set1', 'Qualitative', 3).mpl_colors

    fig = plt.figure()
    stim_strengths, rt, ci = data[:, 0], data[:, 1], data[:, 2]
    k, B0, Bdel, B2, tnd, tnd_sd, stim_strength_bias, y0 = theta

    ustim_strengths = np.unique(stim_strengths)
    stim_strengths_to_propagate = np.linspace(
        np.min(ustim_strengths), np.max(ustim_strengths), 50)

    drifts, t, Bup, Blo, y, yinit = discretize(
        theta, stim_strengths_to_propagate, np.max(rt), dt)

    D = propagate(drifts, t, Bup, Blo, y, yinit)

    meanprop = sortedFunc(ci, stim_strengths, 'mean')[:, 1]
    stdprop = sortedFunc(ci, stim_strengths, 'std')
    semprop = stdprop[:, 1] / np.sqrt(stdprop[:, 2])

    predprop = D['up']['p'][:]

    meanrt_lo = np.zeros(len(ustim_strengths))
    semrt_lo = np.zeros(len(ustim_strengths))
    meanrt_up = np.zeros(len(ustim_strengths))
    semrt_up = np.zeros(len(ustim_strengths))

    ustim_strengths = np.sort(np.unique(stim_strengths))
    for i, stim_strength in enumerate(ustim_strengths):
        s_hi = np.logical_and(stim_strengths == stim_strength, ci == 1)
        s_lo = np.logical_and(stim_strengths == stim_strength, ci == 0)

        meanrt_lo[i] = np.mean(rt[s_lo])
        semrt_lo[i] = np.std(rt[s_lo], ddof=1) / np.sqrt(len(rt[s_lo]))

        meanrt_up[i] = np.mean(rt[s_hi])
        semrt_up[i] = np.std(rt[s_hi], ddof=1) / np.sqrt(len(rt[s_hi]))

    predrt_up = D['up']['mean_t'][0] + tnd
    predrt_lo = D['lo']['mean_t'][0] + tnd

    ax_choice = fig.add_subplot(1, 2, 1)
    ax_choice.errorbar(ustim_strengths, meanprop, yerr=semprop, color='k',
                       linestyle='None', capsize=0, marker='o')
    ax_choice.add_line(
        plt.Line2D(stim_strengths_to_propagate, predprop, color='k'))

    xdata = stim_strengths_to_propagate
    ax_choice.set_xlim((np.min(xdata) * 1.2, np.max(xdata) * 1.2))
    ax_choice.set_ylim((-0.01, 1.01))
    ax_choice.set_xlabel('Stimulus strength')
    ax_choice.set_ylabel('Proportion up responses')

    ax_rt = fig.add_subplot(1, 2, 2)
    ax_rt.errorbar(ustim_strengths, meanrt_up, yerr=semrt_up,
                   color=colors[0], linestyle='None', marker='o', capsize=0)
    ax_rt.errorbar(ustim_strengths, meanrt_lo, yerr=semrt_lo,
                   color=colors[1], linestyle='None', marker='o', capsize=0)
    ax_rt.add_line(plt.Line2D(
        stim_strengths_to_propagate, predrt_up,
        color=colors[0], marker='None'))
    ax_rt.add_line(plt.Line2D(
        stim_strengths_to_propagate, predrt_lo,
        color=colors[1], marker='None'))

    ax_rt.set_xlabel('Stimulus strength')
    ax_rt.set_ylabel('Reaction time (s)')

    ax_rt.set_xlim((np.min(xdata) * 1.2, np.max(xdata) * 1.2))
    ax_rt.set_ylim(np.nanmin(np.concatenate([meanrt_up, meanrt_lo])) * .75,
                   np.nanmax(np.concatenate([meanrt_up, meanrt_lo])) * 1.1)

    ax_rt.set_xlabel('Stimulus strength')
    ax_rt.set_ylabel('Mean reaction times (s)')

    leg = ax_rt.legend(['up RTs', 'lo RTs'])
    for color, text in zip(colors[0:2], leg.get_texts()):
        text.set_color(color)

    sns.despine(offset=5, trim=True)
    plt.tight_layout()


def calculateNegLL(theta, stim_strengths, rt, ci, dt=0.0005):
    """
    Given a  diffusion to bound model with quadratically collapsing bounds
    parametrized by theta, compute the negative log likelihood of the data

    Parameters
    ----------
    theta : 8 element list or array containing the DTB parameters
            k, B0, Bdel, B2, tnd, tnd_sd, stim_strength_bias, y0

    stim_strengths : array of stimulus strengths, length equal to # trials

    rt : array of reaction times, length equal to # trials

    ci : array of choices (0, 1: 1 is correct choice for positive
         stim_strengths), length equal to # trials

    dt : float, temporal resolution, in seconds, to use when propagating

    Returns
    -------
    neglogll : float, negative log likelihood of the data

    """

    global T, It, STIM_STRENGTH, Ic

    # unpack parameters
    k, B0, Bdel, B2, tnd, tnd_sd, stim_strength_bias, y0 = theta

    u_stim_strengths = np.unique(stim_strengths)

    # propagate expects drift to be 2d so that it can tile properly
    drifts, t, Bup, Blo, y, yinit = discretize(
        theta, u_stim_strengths, np.max(rt), dt)

    # propagate density
    D = propagate(drifts, t, Bup, Blo, y, yinit)

    # convolve distribution of exit times with Gaussian distribution of tnd
    r = norm.pdf(t, tnd, tnd_sd) * dt
    P_UP = np.zeros([D['up']['pdf_t'].shape[0] + r.shape[0] - 1,
                     D['up']['pdf_t'].shape[1]])
    P_LO = np.zeros([D['lo']['pdf_t'].shape[0] + r.shape[0] - 1,
                     D['up']['pdf_t'].shape[1]])
    for i in xrange(len(u_stim_strengths)):
        P_UP[:, i] = fftconvolve(D['up']['pdf_t'][:, i], r.flatten())
        P_LO[:, i] = fftconvolve(D['lo']['pdf_t'][:, i], r.flatten())

    if T is None:
        # find time index of each trial
        T = np.tile(t, [1, len(rt)])
        It = np.sum(~(T >= rt), axis=0) - 1

        # find stim_strength index of each trial
        C = np.tile(np.atleast_2d(u_stim_strengths), [len(stim_strengths), 1])
        Ic = np.sum(C <= np.atleast_2d(stim_strengths).T, axis=1) - 1

    p_up = P_UP[It, Ic]
    p_lo = P_LO[It, Ic]
    p_up = np.clip(p_up, eps, 1 - eps)
    p_lo = np.clip(p_lo, eps, 1 - eps)

    ppred = p_up * ci + p_lo * np.logical_not(ci)
    neglogl = -np.sum(np.log(ppred))

    print(
        'NegLL = {:.3f}, k = {:.2f}, B0 = {:.2f}, Bdel = {:.3f}, B2 = {:.4f} '
        .format(neglogl, theta[0], theta[1], theta[2], theta[3]) +
        'tnd = {:.2f}, tnd_sd = {:.2f}, stim_bias = {:.5f}, y0 = {:.5f}'
        .format(theta[4], theta[5], theta[6], theta[7]))

    return neglogl


def propagate(drifts, t, Bup, Blo, y, y0, notabs_flag=False):
    """
    Given a set of drift rates and bounds, compute various quantities
    that characterize the behavior of the drift-diffusion process

    Parameters
    ----------
    drifts : 2D array of shape (1, nd) where nd is the number of drift rates;
             this is a row vector of drifts

    t : 2D array of shape (nt, 1) where nt is the number of time points;
        this is a column vector of time points

    Bup : 2D array of shape (nt, 1) where nt is the number of time points;
          this is a column vector of upper bounds, one value per each time bin

    Blo : 2D array of shape (nt, 1) where nt is the number of time points;
          this is a column vector of lower bounds, one value per each time bin

    y : 2D array of shape (ng, 1) where ng is the number of grid points
        over which to compute Gaussian density; this is a column vector
        of grid points to which y0 corresponds

    y0 : 2D array of shape (ng, 1) where ng is the number of grid points
         over which to compute Gaussian density; this is a column vector
         of actual initial density values

    notabs_flag : boolean, whether to store the not absorbed portion of density

    Returns
    -------
    D : dictionary, contains for each unique stimulus strengh the probability
        of hitting the upper or lower bound, the mean reaction times for the
        two bounds, and the cumulative distribution of reaction times for the
        two bounds (and the not absored densities if requested)

    """

    nd = drifts.shape[1]
    nt = t.shape[0]
    dt = t[1, 0] - t[0, 0]
    ny = y.shape[0]

    D = {}
    D['bounds'] = np.column_stack([Bup, Blo])
    D['drifts'] = drifts

    # create fft of zero mean, unit variance gaussian
    kk = np.tile(np.atleast_2d(np.concatenate(
        [np.arange(0, ny / 2 + 1), np.arange(-ny / 2 + 1, 0)])).T, [1, nd])
    omega = 2 * np.pi * kk / (np.max(y) - np.min(y))
    E1 = np.exp(-0.5 * dt * np.power(omega, 2))
    D['up'] = {}
    D['lo'] = {}
    if notabs_flag:
        D['notabs'] = {}

    D['up']['pdf_t'] = np.zeros([nt, nd])
    D['lo']['pdf_t'] = np.zeros([nt, nd])
    if notabs_flag:
        D['notabs']['pdf'] = np.zeros([nd, ny, nt])

    # U is initial state, one column for each drift rate
    U = np.tile(y0, [1, nd])

    # shift mean of zero mean, unit variance gaussian by drift
    E2 = E1 * np.exp(-1j * omega * np.tile(drifts, [ny, 1]) * dt)

    # prepare the values to propagate for vectorization
    y = np.tile(y, [1, nd])
    p_threshold = .00001
    for i in range(nt):
        # fft current pdf
        Ufft = np.fft.fft(U, axis=0)

        # convolve with gaussian with drift in the frequency domain
        Ufft = E2 * Ufft

        # turn back into time domain
        U = np.real(np.fft.ifft(Ufft, axis=0))

        D['up']['pdf_t'][i, :] = np.sum(U * (y >= Bup[i]), axis=0)
        D['lo']['pdf_t'][i, :] = np.sum(U * (y <= Blo[i]), axis=0)

        # keep only density within bounds
        U *= np.logical_and(y > Blo[i], y < Bup[i])

        # store not absorbed density if requested
        if notabs_flag:
            D['notabs']['pdf'][:, :, i] = U.T

        if np.sum(np.sum(U, axis=0) < p_threshold) == nd:
            break

    D['t'] = t

    D['up']['p'] = np.sum(D['up']['pdf_t'], axis=0)
    D['lo']['p'] = np.sum(D['lo']['pdf_t'], axis=0)

    D['up']['mean_t'] = np.dot(t.T, D['up']['pdf_t']) / D['up']['p']
    D['lo']['mean_t'] = np.dot(t.T, D['lo']['pdf_t']) / D['lo']['p']

    D['up']['cdf_t'] = np.cumsum(D['up']['pdf_t'], axis=0)
    D['lo']['cdf_t'] = np.cumsum(D['lo']['pdf_t'], axis=0)

    return D


def discretize(theta, stim_strengths, tmax, dt):
    """
    Utility function that sets up the grid on which to do density propagation
    and also computes some additional required quantities

    Parameters
    ----------
    theta : 8 element list or array containing the DTB parameters
            k, B0, Bdel, B2, tnd, tnd_sd, stim_strength_bias, y0

    stim_strengths : array of unique stimulus strength for which to do
                     propagation

    tmax : float, maximum time for which to propagate

    dt : float, temporal resolution, in seconds, to use when propagating

    Returns
    -------
    drifts : array of shape (nd,) when nd is the number of drifts

    t : array of shape (nt,) where nt is the number of time points;
        vector of time points

    Bup : array of shape (nt,) where nt is the number of time points;
          vector of upper bounds, one value per each time bin

    Blo : array of shape (nt,) where nt is the number of time points;
          vector of lower bounds, one value per each time bin

    y : array of shape (ng,) where ng is the number of grid points
        over which to compute Gaussian density;
        these are the grid points to which y0 corresponds

    yinit : array of shape (ng,) where ng is the number of grid points
            over which to compute Gaussian density;
            these are the  actual initial density values

    """

    k, B0, Bdel, B2, tnd, tnd_sd, stim_strength_bias, y0 = theta

    # time
    tmax = tmax + .3
    t = np.arange(0, tmax, dt)

    # bounds
    Bup = B0 * (t == t)
    s = t > Bdel
    Bup[s] = B0 - B2 * np.power(t[s] - Bdel, 2)
    Blo = -Bup

    # drift rate
    drifts = k * (stim_strengths + stim_strength_bias)

    md = np.max(np.abs(drifts))
    sm = md * dt + np.sqrt(dt) * 4
    y = np.linspace(np.min(Blo) - sm, np.max(Bup) + sm, 512)
    yinit = 0 * y

    i1 = np.where(y >= y0)[0][0]
    i2 = np.where(y <= y0)[0][-1]

    if i1 == i2:
        yinit[i1] = 1
    else:
        w2 = np.abs(y[i1] - y0)
        w1 = np.abs(y[i2] - y0)

        w1 = w1 / (w1 + w2)
        w2 = (1 - w1)
        yinit[i1] = w1
        yinit[i2] = w2

    drifts = drifts[np.newaxis, :]
    t = t.reshape(-1, 1)
    Bup = Bup.reshape(-1, 1)
    Blo = Blo.reshape(-1, 1)
    y = y.reshape(-1, 1)
    yinit = yinit.reshape(-1, 1)

    return drifts, t, Bup, Blo, y, yinit


def sortedFunc(x, y, func):
    unique_y = np.sort(np.unique(y))
    ny = len(unique_y)

    out_x = np.empty(ny)
    counts_y = np.empty(ny)
    for i, j in zip(range(ny), unique_y):
        out_x[i] = getattr(np, func)(np.array(x)[np.array(y) == j])
        counts_y[i] = np.sum(np.array(y) == j)

    return(np.column_stack([unique_y, out_x, counts_y]))


if __name__ == '__main__':
    theta = [12.5, 0.6, 0.3, 0.3, 0.4, 0.05, 0.0, 0.0]

    print('\n')
    print('Generating data')
    print('\n')
    data = simulateData(*theta)

    print('\n')
    print('Fitting this synthetic dataset will take a few mins, be patient')
    print('\n')
    fitted_theta = fitData(data, dt=0.0005)

    print('---' * 20)
    print('---' * 20)
    print('True k: {:.3f}, Fitted k: {:.3f}'.format(
        theta[0], fitted_theta[0]))
    print('True B0: {:.3f}, Fitted B0: {:.3f}'.format(
        theta[1], fitted_theta[1]))
    print('True Bdel: {:.3f}, Fitted Bdel: {:.3f}'.format(
        theta[2], fitted_theta[2]))
    print('True B2: {:.3f}, Fitted B2: {:.3f}'.format(
        theta[3], fitted_theta[3]))
    print('True tnd: {:.3f}, Fitted tnd: {:.3f}'.format(
        theta[4], fitted_theta[4]))
    print('True tnd_sd: {:.3f}, Fitted tnd_sd: {:.3f}'.format(
        theta[5], fitted_theta[5]))
    print('True bias: {:.4f}, Fitted bias: {:.4f}'.format(
        theta[6], fitted_theta[6]))
    print('True offset: {:.4f}, Fitted offset: {:.4f}'.format(
        theta[7], fitted_theta[7]))
    print('---' * 20)
    print('---' * 20)
