"""
https://en.wikipedia.org/wiki/Order_statistic
http://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/notes/noteorder.pdf
"""

from math import factorial
import scipy.interpolate


def calc_dist_val_k_os(sample_space, dist, cdf, is_discrete, k, n, x):
    if is_discrete:
        f = 0.0
    else:
        interp_dist = scipy.interpolate.interp1d(sample_space, dist)
        interp_cdf = scipy.interpolate.interp1d(sample_space, cdf)
        f = factorial(n) / (factorial(k - 1) * factorial(n - k))
        f *= interp_cdf(x) ** (k - 1)
        f *= (1.0 - interp_cdf(x)) ** (n - k)
        f *= interp_dist(x)
    return f


def calc_joint_dist_val_jk_os(sample_space, dist, cdf, is_discrete, j, k, n, x, y):
    if is_discrete:
        f = 0.0
    else:
        interp_dist = scipy.interpolate.interp1d(sample_space, dist)
        interp_cdf = scipy.interpolate.interp1d(sample_space, cdf)
        f = factorial(n) / (factorial(j - 1) * factorial(k - j - 1) * factorial(n - k))
        f *= interp_cdf(x) ** (j - 1)
        f *= (interp_cdf(y) - interp_cdf(x)) ** (k - 1 - j)
        f *= (1 - interp_cdf(y)) ** (n - k)
        f *= interp_dist(x) * interp_dist(y)
    return f
