import json
import copy
import math

class Gaus1D(object):
    def __init__(self, mean, variance, prior_strength=2):
        """
        Initialize a 1D gaussian. Assume it was generated from one data point
        :param mean - the center of the gaussian, floating point
        :param variance - the variance of the gaussian, floating point, =
            std_dev ** 2
        :param n - the number of points assumed to have 'made' the gaussian
        """
        self.dist_type = __class__.__name__
        assert variance != 0, '{} cannot be created with zero' \
            'variance'.format(self.dist_type)
        assert prior_strength >= 2, "Prior strength must be 2 or larger"
        self.mean = mean
        self.variance = variance
        self.n = prior_strength
        self.M2 = self.variance * (self.n - 1) 

    def set_prior_strength(self,prior_strength):
        assert prior_strength >= 2, "Prior strength must be 2 or larger"
        self.n = prior_strength
        self.M2 = self.variance * (self.n-1)

    def to_json_string(self):
        return json.dumps(self.__dict__)

    def update(self, point, debug = False):
        """
        Refit gaussian object to a new point.
        """
        self.n += 1
        delta = point - self.mean
        self.mean += delta/self.n
        self.M2 += delta*(point - self.mean)
        if debug: print(self.variance)
        self.variance = self.M2/(self.n-1)
        if debug: print(self.variance)
        if debug: print("="*100)
        # TODO: handle this intelligently (???)
        if self.variance==0: 
            assert self.variance != 0, '{} is updating itself to have zero' \
                'variance, which is not allowed.' \
                .format(self.dist_type)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return json.dumps(self.__dict__,indent=2)

    def __mul__(self,g2):
        if not isinstance(g2, Gaus1D):
            raise ValueError("{}.{} needs argument of type {}. " \
                    "Given argument: {}".format(
                        self.dist_type, 
                        self.__mul__.__name__, 
                        self.dist_type,
                        type(g2)
                        )
                    )
        if self.variance == 0 or g2.variance == 0:
            raise ValueError("{} objects cannot have zero variance. " \
                    "SELF variance: {}, ARG variance: {}".format(
                        self.dist_type, 
                        self.variance, 
                        g2.variance)
                    )

        variance_3 = (self.variance**-1. + g2.variance**-1)**-1
        mean_3 = (variance_3 * self.variance**-1.*self.mean + variance_3 * g2.variance**-1 * g2.mean)
        return_gaus = self.copy()
        return_gaus.variance = variance_3
        return_gaus.mean = mean_3
        return_gaus.M2 = None
        return_gaus.n = None
        return return_gaus

    def __rmul__(self,g2):
        return self.__mul__(g2)

    def KL_Div(self,g2):
        """
        Calculates KL divergence between self and another gaussian
        distribution.

        Assume self is the LHS of KL operation and g2 is RHS
        """
        if not isinstance(g2, Gaus1D):
            raise ValueError("{}.{} needs argument of type {}. " \
                "Given argument: {}".format(
                    self.dist_type,
                    self.__mul__.__name__,
                    self.dist_type,
                    type(g2)
                    )
                )
        sig_1 = self.variance**0.5
        sig_2 = g2.variance**0.5
        mu_1 = self.mean
        mu_2 = g2.mean

        KL = math.log(sig_2/sig_1) + (sig_1**2+(mu_1-mu_2)**2)/(2*sig_2**2) - 0.5
        return KL

def gaus_1D_from_json(json_string):
    '''Create a Gaus1D object from a json string'''
    try:
        obj = json.loads(json_string)
        g = Gaus1D(0,1,2)
        g.mean = float(obj['mean'])
        g.variance = float(obj['variance'])
        g.M2 = float(obj['M2'])
        g.n = float(obj['n'])
        g.dist_type = str(obj["dist_type"])
    except:
        raise ValueError('{} did not map to class structure: {}'.format(json_string,Gaus1D(0,1,2).to_json_string()))
    return g.copy()

if __name__ == '__main__':
    # Unit tests
    print(repr(Gaus1D(2.0,2.0**2)))
    g1 = Gaus1D(2.0,2.0**2)
    g2 = Gaus1D(2.0,4.0**2)

    print(g1.__dict__)
    print('g2',g2)
    mystr = g2.to_json_string()
    g1 = gaus_1D_from_json(mystr).update(3.0)
    print(g1)
