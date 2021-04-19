import numpy as np
from math import pi
from simsopt.core.optimizable import Optimizable
from simsopt.geo.curve import RotatedCurve


class CoilCollection(Optimizable):
    """
    Given some input coils and currents, this performs the reflection and
    rotation to generate a full set of stellarator coils.
    """

    def __init__(self, coils, currents, nfp, stellarator_symmetry):
        self._base_coils = coils
        self._base_currents = np.asarray(currents)
        self.coils = []
        self.currents = []
        flip_list = [False, True] if stellarator_symmetry else [False]
        self.map = []
        self.current_sign = []
        for k in range(0, nfp):
            for flip in flip_list:
                for i in range(len(self._base_coils)):
                    if k == 0 and not flip:
                        self.coils.append(self._base_coils[i])
                        self.currents.append(self._base_currents[i])
                    else:
                        rotcoil = RotatedCurve(coils[i], 2*pi*k/nfp, flip)
                        current = -currents[i] if flip else currents[i]
                        self.coils.append(rotcoil)
                        self.currents .append(current)
                    self.map.append(i)
                    self.current_sign.append(-1 if flip else +1)
        self.depends_on = self._base_coils

    # def set_dofs(self, dofs):
    #     self._base_currents[:] = dofs[:]
    #     self.currents = [self.current_sign[i]*self._base_currents[self.map[i]] for i in range(len(dofs))]

    # def get_dofs(self):
    #     return self._base_currents

    # def num_dofs(self):
    #     return len(self._base_currents)

    def reduce_coefficient_derivatives(self, derivatives, axis=0):
        """
        Add derivatives for all those coils that were obtained by rotation and
        reflection of the initial coils.
        """
        assert len(derivatives) == len(self.coils) or len(derivatives) == len(self._base_coils)
        res = len(self._base_coils) * [None]
        for i in range(len(derivatives)):
            if res[self.map[i]] is None:
                res[self.map[i]]  = derivatives[i]
            else:
                res[self.map[i]] += derivatives[i]
        return np.concatenate(res, axis=axis)

    def reduce_current_derivatives(self, derivatives):
        """
        Combine derivatives with respect to current for all those coils that
        were obtained by rotation and reflection of the initial coils.
        """
        assert len(derivatives) == len(self.coils) or len(derivatives) == len(self._base_coils)
        res = len(self._base_coils) * [None]
        for i in range(len(derivatives)):
            if res[self.map[i]] is None:
                res[self.map[i]]  = self.current_sign[i] * derivatives[i]
            else:
                res[self.map[i]] += self.current_sign[i] * derivatives[i]
        return np.asarray(res)
