import math

""" BSM model:
    c0 = S0 * phi(d1)   -   e^(-rT) * K * phi(d2)
    p0 = e^(-rT) * K * phi(-d2)  -   S0 * phi(-d1)
    d1 = [log(S0/K) + (r+sigma^2)T] / [sigma * sqrt(T)]
    d2 = d1 - [sigma * sqrt(T)] """


def phi(x):
    # cdf for the standard normal distribution: P(Z<=x)
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def small_phi(x):
    return math.exp(-(x*x)/2) / math.sqrt(2 * math.pi)


def bsm_d1_d2(s0, k, t, sigma, r):
    sigma_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s0/k) + (r + (sigma * sigma)/2)*t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


def bsm_eu_call(s0, k, t, sigma, r, d_pv=0):
    s0 -= d_pv
    d1, d2 = bsm_d1_d2(s0, k, t, sigma, r)
    return s0 * phi(d1) - math.exp(-r*t) * k * phi(d2)


def bsm_eu_put(s0, k, t, sigma, r, d_pv=0):
    s0 -= d_pv
    d1, d2 = bsm_d1_d2(s0, k, t, sigma, r)
    return math.exp(-r*t) * k * phi(-d2) - s0 * phi(-d1)


def itm_probability_eu_call(s0, k, t, sigma, r, d_pv=0):
    """ Risk neutral probability that call option is in-the-money """
    s0 -= d_pv
    d1, d2 = bsm_d1_d2(s0, k, t, sigma, r)
    return phi(d2)


def itm_probability_eu_put(s0, k, t, sigma, r, d_pv=0):
    """ Risk neutral probability that put option is in-the-money """
    s0 -= d_pv
    d1, d2 = bsm_d1_d2(s0, k, t, sigma, r)
    return phi(-d2)


def bsm_expected_maturity_price_call(s0, k, t, sigma, r, d_pv=0):
    """ Risk neutral expectation of stock price at T given that it is in-the-money E[S_T | S_T > K] """
    s0 -= d_pv
    d1, d2 = bsm_d1_d2(s0, k, t, sigma, r)
    return math.exp(r*t) * s0 * phi(d1) / phi(d2)


def bsm_expected_maturity_price_put(s0, k, t, sigma, r, d_pv=0):
    """ Risk neutral expectation of stock price at T given that it is in-the-money E[S_T | S_T < K]"""
    s0 -= d_pv
    d1, d2 = bsm_d1_d2(s0, k, t, sigma, r)
    return math.exp(r * t) * s0 * phi(-d1) / phi(-d2)


def bsm_expected_maturity_profit_call(s0, k, t, sigma, r, d_pv=0):
    """ Risk neutral expectation of profit at T given that it is in-the-money E[S_T - K | S_T > K]"""
    s0 -= d_pv
    return bsm_expected_maturity_price_call(s0, k, t, sigma, r) - k


def bsm_expected_maturity_profit_put(s0, k, t, sigma, r, d_pv=0):
    """ Risk neutral expectation of profit at T given that it is in-the-money E[K - S_T | S_T < K]"""
    s0 -= d_pv
    return k - bsm_expected_maturity_price_call(s0, k, t, sigma, r)


def time_value_call(s0, k, t, sigma, r, d_pv=0):
    c = bsm_eu_call(**locals())
    return c - s0 + k


def time_value_put(s0, k, t, sigma, r, d_pv=0):
    p = bsm_eu_put(**locals())
    return p - k + s0


def dividends_pv(d_df, r):
    d_df['PV'] = [math.exp(-r * x) * y for x, y in tuple(zip(d_df['T'], d_df['D']))]
    return d_df['PV'].sum()


def bsm_american_call(s0, k, t, sigma, r, d_df=None):
    """ Black's Approximation: C0 = Max(c0, k0), k0 is european call value expired before the last dividend """
    if d_df is not None:
        sum_d_pv_til_last = dividends_pv(d_df.iloc[:-1, :], r)
        early_exercise_t = d_df['T'].iloc[-1]
        early_exercise_value = bsm_eu_call(s0, k, early_exercise_t, sigma, r, d_pv=sum_d_pv_til_last)
        eu_call_value = bsm_eu_call(s0,k,t,sigma,r,d_pv=dividends_pv(d_df,r))
        return max((early_exercise_value,eu_call_value))
    else:
        return bsm_eu_call(s0, k, t, sigma, r)


def bs_equation(s, sigma, v, r, theta=None, delta=None, gamma=None):
    known_param = sum((1 if x is not None else 0 for x in (theta, delta, gamma)))
    if known_param != 2:
        raise Exception('At least 2 known greeks')
    else:
        if theta is None:
            theta = r*v - s*s*sigma*sigma*gamma/2 - delta*r*s
        elif delta is None:
            delta = (r*v - s*s*sigma*sigma*gamma/2 - theta)/(r*s)
        else:
            gamma = (r*v - theta - delta*r*s)*2 / (sigma*sigma*s*s)
    return theta, delta, gamma

