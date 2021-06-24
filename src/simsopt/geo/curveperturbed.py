import numpy as np
import simsoptpp as sopp
from .curve import Curve
try:
    from sympy import Symbol, lambdify, exp
    sympy_found = True
except ImportError:
    sympy_found = False


class GaussianSampler():

    def __init__(self, points, sigma, length_scale, n_derivs=1):
        self.points = points
        xs = self.points
        n = len(xs)
        self.n_derivs = n_derivs
        cov_mat = np.zeros((n*(n_derivs+1), n*(n_derivs+1)))
        def kernel(x, y):
            return sum(sigma**2*exp(-(x-y+i)**2/length_scale**2) for i in range(-2, 3))
        for ii in range(n_derivs+1):
            for jj in range(n_derivs+1):
                if ii + jj == 0:
                    lam = kernel
                else:
                    x = Symbol("x")
                    y = Symbol("y")
                    f = kernel(x, y)
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])))
                for i in range(n):
                    for j in range(n):
                        x = xs[i]
                        y = xs[j]
                        if abs(x-y)>0.5:
                            if y>0.5:
                                y -= 1
                            else:
                                x -= 1
                        cov_mat[ii*n + i, jj*n + j] = lam(x, y)

        from scipy.linalg import sqrtm
        self.L = np.real(sqrtm(cov_mat))

    def sample(self, randomgen=None):
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


class CurvePerturbed(sopp.Curve, Curve):

    def __init__(self, curve, sampler, randomgen=None):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self)
        self.sampler = sampler
        self.randomgen = randomgen
        self.sample = sampler.sample(self.randomgen)
        curve.dependencies.append(self)

    def resample(self):
        self.sample = self.sampler.sample(self.randomgen)
        self.invalidate_cache()

    def num_dofs(self):
        return self.curve.num_dofs()

    def get_dofs(self):
        return self.curve.get_dofs()

    def set_dofs_impl(self, x):
        return self.curve.set_dofs(x)

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.curve.gamma() + self.sample[0]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        dgamma_by_dcoeff[:] = self.curve.dgamma_by_dcoeff()

    def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
        dgammadash_by_dcoeff[:] = self.curve.dgammadash_by_dcoeff()

    def dgammadashdash_by_dcoeff_impl(self, dgammadashdash_by_dcoeff):
        dgammadashdash_by_dcoeff[:] = self.curve.dgammadashdash_by_dcoeff()

    def dgammadashdashdash_by_dcoeff_impl(self, dgammadashdashdash_by_dcoeff):
        dgammadashdashdash_by_dcoeff[:] = self.curve.dgammadashdashdash_by_dcoeff()

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)
