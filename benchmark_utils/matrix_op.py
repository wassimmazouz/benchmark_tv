import numpy as np


def grad_rgb(u):
    gh = np.zeros_like(u)
    gv = np.zeros_like(u)

    for c in range(3):
        # Apply the gradient calculation to each channel separately
        gh[c] = np.pad(np.diff(u[c], axis=0), ((0, 1), (0, 0)), 'constant')
        gv[c] = np.pad(np.diff(u[c], axis=1), ((0, 0), (0, 1)), 'constant')

    return gh, gv
