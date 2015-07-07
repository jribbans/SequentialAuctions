"""
https://en.wikipedia.org/wiki/Order_statistic
http://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/notes/noteorder.pdf
"""

from math import factorial
import scipy.interpolate


def calc_dist_val_k_os(sample_space, dist, cdf, is_discrete, k, n, x):
    assert not is_discrete, "Only continuous distributions are supported."
    if is_discrete:
        pass
    else:
        interp_dist = scipy.interpolate.interp1d(sample_space, dist)
        interp_cdf = scipy.interpolate.interp1d(sample_space, cdf)
        f = factorial(n) / (factorial(k - 1) * factorial(n - k))
        f *= interp_cdf(x) ** (k - 1.0)
        f *= (1.0 - interp_cdf(x)) ** (n - k)
        f *= interp_dist(x)
    return f


def calc_dist_val_k_os_u01(is_discrete, k, n, x):
    assert not is_discrete, "Only continuous distributions are supported."
    if is_discrete:
        pass
    else:
        f = factorial(n) / (factorial(k - 1) * factorial(n - k))
        f *= x ** (k - 1)
        f *= (1.0 - x) ** (n - k)
    return f


def calc_joint_dist_val_jk_os(sample_space, dist, cdf, is_discrete, j, k, n, x, y):
    assert not is_discrete, "Only continuous distributions are supported."
    if is_discrete:
        pass
    else:
        interp_dist = scipy.interpolate.interp1d(sample_space, dist)
        interp_cdf = scipy.interpolate.interp1d(sample_space, cdf)
        f = factorial(n) / (factorial(j - 1) * factorial(k - j - 1) * factorial(n - k))
        f *= interp_cdf(x) ** (j - 1)
        f *= (interp_cdf(y) - interp_cdf(x)) ** (k - 1 - j)
        f *= (1 - interp_cdf(y)) ** (n - k)
        f *= interp_dist(x) * interp_dist(y)
    return f


def calc_joint_dist_val_jk_os_u01(is_discrete, j, k, n, x, y):
    assert not is_discrete, "Only continuous distributions are supported."
    if is_discrete:
        pass
    else:
        f = factorial(n) / (factorial(j - 1) * factorial(k - j - 1) * factorial(n - k))
        f *= x ** (j - 1)
        f *= (y - x) ** (k - 1 - j)
        f *= (1 - y) ** (n - k)
    return f
