""" This module reproduces scipy.stats.truncnorm as of SciPy v1.3.3. More recent versions of truncnorm are orders of magnitude slower
as documented here and elsewhere: https://github.com/scipy/scipy/issues/12370.
"""

import numpy as np
from scipy.stats._continuous_distns import rv_continuous, _norm_cdf, _norm_sf, _norm_pdf, _norm_logpdf, _norm_isf, _norm_ppf

class truncnorm_gen(rv_continuous):
    def _argcheck(self, a, b):
        return a < b

    def _get_support(self, a, b):
        return a, b

    def _get_norms(self, a, b):
        _nb = _norm_cdf(b)
        _na = _norm_cdf(a)
        _sb = _norm_sf(b)
        _sa = _norm_sf(a)
        _delta = np.where(a > 0, _sa - _sb, _nb - _na)
        with np.errstate(divide='ignore'):
            return _na, _nb, _sa, _sb, _delta, np.log(_delta)

    def _pdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _delta = ans[4]
        return _norm_pdf(x) / _delta

    def _logpdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _logdelta = ans[5]
        return _norm_logpdf(x) - _logdelta

    def _cdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _na, _delta = ans[0], ans[4]
        return (_norm_cdf(x) - _na) / _delta

    def _ppf(self, q, a, b):
        # XXX Use _lazywhere...
        ans = self._get_norms(a, b)
        _na, _nb, _sa, _sb = ans[:4]
        ppf = np.where(a > 0,
                       _norm_isf(q*_sb + _sa*(1.0-q)),
                       _norm_ppf(q*_nb + _na*(1.0-q)))
        return ppf

    def _stats(self, a, b):
        ans = self._get_norms(a, b)
        nA, nB = ans[:2]
        d = nB - nA
        pA, pB = _norm_pdf(a), _norm_pdf(b)
        mu = (pA - pB) / d   # correction sign
        mu2 = 1 + (a*pA - b*pB) / d - mu*mu
        return mu, mu2, None, None

truncnorm = truncnorm_gen(name='truncnorm')