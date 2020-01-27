import math
from models import bsm


def delta(s0, k, t, sigma, r, d_pv=0, is_call=True):
    """ partial derivative of C(S,K,sigma,r,t) with respect to S """
    s0 -= d_pv
    d1, d2 = bsm.bsm_d1_d2(s0, k, t, sigma, r)
    if is_call:
        return bsm.phi(d1)
    else:
        return bsm.phi(d1) - 1


def lambda_greek(s0, k, t, sigma, r, d_pv=0, is_call=True):
    """ leverage ratio: delta*(S/V) , for 1% change in stock price, lambda % change in option value"""
    d = delta(**locals())
    if is_call:
        v = bsm.bsm_eu_call(s0, k, t, sigma, r, d_pv)
    else:
        v = bsm.bsm_eu_put(s0, k, t, sigma, r, d_pv)
    return d * s0 / v


def theta(s0, k, t, sigma, r, d_pv=0, is_call=True):
    """ partial derivative of C(S,K,sigma,r,t) with respect to T """
    s0 -= d_pv
    d1, d2 = bsm.bsm_d1_d2(s0, k, t, sigma, r)
    if is_call:
        return -s0*bsm.small_phi(d1)*sigma/(2*math.sqrt(t)) - r*k*math.exp(-r*t)*bsm.phi(d2)
    else:
        return -s0*bsm.small_phi(d1)*sigma/(2*math.sqrt(t)) + r*k*math.exp(-r*t)*bsm.phi(-d2)


def gamma(s0, k, t, sigma, r, d_pv=0):
    """ second partial derivative of C(S,K,sigma,r,t) with respect to S, dDelta/dS"""
    s0 -= d_pv
    d1, d2 = bsm.bsm_d1_d2(s0, k, t, sigma, r)
    return bsm.small_phi(d1)/(s0*sigma*math.sqrt(t))


def vega(s0, k, t, sigma, r, d_pv=0):
    """ partial derivative of C(S,K,sigma,r,t) with respect to sigma """
    s0 -= d_pv
    d1, d2 = bsm.bsm_d1_d2(s0,k,t,sigma, r)
    return s0*math.sqrt(t)*bsm.small_phi(d1)


def rho(s0, k, t, sigma, r, d_pv=0, is_call=True):
    """ partial derivative of C(S,K,sigma,r,t) with respect to r """
    s0 -= d_pv
    d1, d2 = bsm.bsm_d1_d2(s0,k,t,sigma,r)
    if is_call:
        return k*t*math.exp(-r*t)*bsm.phi(d2)
    else:
        return -k*t*math.exp(-r*t)*bsm.phi(-d2)

