# Bounded fminsearch
#
# Written for Matlab by John D'Errico
# Translated into Python by Luke Woloszyn
#

import numpy as np
from scipy.optimize import fmin


def fminbnd(
        func, x0, args=(), LB=None, UB=None,
        disp=False, xtol=1e-4, ftol=1e-4,
        maxiter=None, maxfun=None):
    n = len(x0)

    if LB is None:
        LB = np.repeat(-np.inf, n)

    if UB is None:
        UB = np.repeat(np.inf, n)

    if (n != len(LB) | n != len(UB)):
        raise RuntimeError('x0 is incompatible in length with either LB or UB')

    # cram stuff into dictionary object
    params = {}
    params['func'] = func
    params['LB'] = LB
    params['UB'] = UB
    params['n'] = n
    params['BoundClass'] = np.zeros(n)

    # 0 --> unconstrained variable
    # 1 --> lower bound only
    # 2 --> upper bound only
    # 3 --> dual finite bounds
    # 4 --> fixed variable
    for i in range(n):
        k = np.isfinite(LB[i]) + 2 * np.isfinite(UB[i])
        params['BoundClass'][i] = k
        if (k == 3) & (LB[i] == UB[i]):
            params['BoundClass'][i] = 4

    # transform starting values into their unconstrained surrogates.
    # check for infeasible starting guesses.
    x0u = x0
    k = 0
    for i in xrange(n):

        # lower bound only
        if params['BoundClass'][i] == 1:
            if x0[i] <= LB[i]:
                # infeasible starting value; use bound
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(x0[i] - LB[i])

            # increment k
            k += 1

        # upper bound only
        elif params['BoundClass'][i] == 2:
            if x0[i] >= UB[i]:
                # infeasible starting value; use bound
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(UB[i] - x0[i])

            # increment k
            k += 1

        # lower and upper bound
        elif params['BoundClass'][i] == 3:
            if x0[i] <= LB[i]:
                # infeasible start value
                x0u[k] = -np.pi / 2
            elif x0[i] >= UB[i]:
                # infeasible starting value
                x0u[k] = np.pi / 2
            else:
                x0u[k] = 2 * (x0[i] - LB[i]) / (UB[i] - LB[i]) - 1
                # shift by 2* pi to avoid problems at zero in fmin;
                # otherwise, the initial simplex is vanishingly small
                x0u[k] = 2 * np.pi + \
                    np.arcsin(np.max([-1, np.min([1, x0u[k]])]))

            # increment k
            k += 1

        # fixed variable. drop it before fmin sees it; k is not incremented
        elif params['BoundClass'][i] == 4:
            continue

        # unconstrained variable
        elif params['BoundClass'][i] == 0:
            x0u[k] = x0[i]
            k += 1

    # if any of the unknowns were fixed, then we need to shorten x0u now
    if k < n:
        x0u[k:n] = []

    # now we can call fminsearch, but with our own intra-objective function
    xu, fopt, iterations, funcalls, warnflag = fmin(
        intrafun, x0u, (params, args),
        full_output=True, disp=disp, xtol=xtol, ftol=ftol,
        maxiter=maxiter, maxfun=maxfun)

    # undo the variable trasnformations into the original space
    x = xtransform(xu, params)

    return x, fopt


def intrafun(x, *params_and_args):
    # transform variables, then call original function
    params, args = params_and_args

    # transform
    xtrans = xtransform(x, params)

    # and call func
    fval = params['func'](xtrans, *args)

    return fval


def xtransform(x, params):
    # converts unconstrained variables into their original domains

    xtrans = np.zeros(params['n'])

    # k allows some variables to be fixed, thus dropped from the optimization
    k = 0

    for i in xrange(params['n']):
        # lower bound only
        if params['BoundClass'][i] == 1:
            xtrans[i] = params['LB'][i] + x[k] ** 2
            k += 1
        # upper bound only
        elif params['BoundClass'][i] == 2:
            xtrans[i] = params['UB'][i] - x[k] ** 2
            k += 1
        # lower and upper bounds
        elif params['BoundClass'][i] == 3:
            xtrans[i] = (np.sin(x[k]) + 1) / 2
            xtrans[i] = xtrans[i] * \
                (params['UB'][i] - params['LB'][i]) + params['LB'][i]
            xtrans[i] = np.max(
                [params['LB'][i], np.min([params['UB'][i], xtrans[i]])])
            k += 1
        # fixed variable, bounds are equal, set it at either bound
        elif params['BoundClass'][i] == 4:
            xtrans[i] = params['LB'][i]
        # unconstrained variable
        elif params['BoundClass'][i] == 0:
            xtrans[i] = x[k]
            k += 1
    return xtrans
