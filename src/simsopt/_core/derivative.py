class Derivative():

    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        x = self.data
        y = other.data
        z = {}
        for k in x.keys():
            z[k] = x[k]
        for k in y.keys():
            if k in z:
                z[k] += y[k]
            else:
                z[k] = y[k]

        return Derivative(z)

