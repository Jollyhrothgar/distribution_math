import json
import copy
import math

class Multinomial(object):
    def __init__(self, size=1, prior_strength=1.):
        """
        Set up a multinomial distribution with a discrete number of classes

        internal structure generated will be:
        n_0 -> counts for 0th class
        n_1 -> counts for 1st class
        ...
        n_n -> counts for ( n == size-1) class.

        Similarly, we have:
        p_0 -> probability of class 0
        p_1 -> probability of class 1
        ...
        p_n -> probability of class n == size - 1 class

        By default, selection will be done by integer, however, if name_classes
        is called, then it will be assumed that the user keeps track of names,
        and this class will just try to look up the keys.

        prior_strength is a weighted value which represents the confidence in
        the prior which the Multinomial represents.
        """
        if not isinstance(size,int) or size < 1:
            raise ValueError('Must initialize {} with an integer ' \
                    'larger than 1'.format(self.dist_type))

        self.dist_type = self.__class__.__name__
        self._n = {n : prior_strength for n in range(size)}
        self._p = {n : 1./float(size) for n in range(size)}
        self.size = size
        self.labeled = False
        self.labels = {}  # you have exactly one chance to label data

    def __repr__(self):
        return json.dumps(self.__dict__,indent=2)

    def copy(self):
        return copy.deepcopy(self)

    def reset_labels(self):
        """
        Reset labeling
        """
        self.labels = {}
        self.labeled = False

    def relabel(self,labels):
        """
        Takes a dictionary of labels {"label":element} such that classes can be called
        with a unique label "label", to refer to an integer element.
        """
        if not isinstance(labels,dict):
            raise ValueError("Must use a dictionary of {'label':integer} to relabel classes of Multinomial" )
        if len(labels) != self.size:
            raise ValueError("Must have the same number of elements as labels in Multinomial. There are {} elements and {} labels.".format(self.size, len(labels)))
        observed_values = {}
        observed_labels = {}
        for k,v in labels.items():
            if not isinstance(v,int) or v not in range(self.size) or isinstance(k,int) or k in observed_labels or v in observed_values:
                print (k, v)
                raise ValueError("Labels must uniquely map to integer which is contained in Multinomial classes.")
            observed_labels[k] = None
            observed_values[v] = None
        self.labels = labels
        self.labeled = True

    def _label_to_index(self,i):
        """
        Filter a potential label into a lookup index
        """
        lookup_i = i
        if self.labeled == True:
            try:
                lookup_i = self.labels[i]
            except:
                KeyError("No label {} exists for Multinomial".format(i))
        if lookup_i not in range(self.size):
        return lookup_i

    def _set_p(self,i,p):
        """
        Sets probability of class i to p. No input checking, so don't shoot your
        foot off.
        """
        index = self._label_to_index(i)
        self._p[index] = p

    def _set_n(self,i,n):
        """
        Sets weight associated with probability. Better use even weighting
        across all classes, but since you're not required to do this with
        checking, don't shoot your foot off.
        """
        index = self._label_to_index(i)
        self._n[index] = n

    def get_max_p(self):
        """
        Returns the indicies of _p with the maximum probability. If several
        keys have the same value, then a list of those keys are returned.
        """
        return [k for k,v in self._p.items() if v == max(self._p.values())]

    def get_max_n(self):
        """
        Returns the indicies of _n with the maximum probability. If several
        keys have the same value, then a list of those keys are returned.
        """
        return [k for k,v in self._n.items() if v == max(self._n.values())]

    def get_p(self,class_label):
        """
        Returns ith class probability
        """
        index = self._label_to_index(class_label)
        return self._p[index]

    def get_n(self, class_label):
        """
        Returns nth class probability
        """
        index = self._label_to_index(class_label)
        return self._n[index]

    def __mul__(self,b1):
        if not isinstance(b1,Multinomial):
            raise ValueError("Must multiply two Multinomial type objects. " \
                    "Type of argument is: {}".format(type(b1)))
        if self.size != b1.size:
            raise ValueError("Multinomial multiplication must be done on " \
                    "distributions of the same dimension. The two dimensions "\
                    "were {} and {}".format(self.size,b1.size))
        
        b_new = self.copy()
        for i in range(self.size):
            b_new._p[i] = self.get_p(i)*b1.get_p(i)
            b_new._set_n(i,None)

        norm = 0.
        for index,p in b_new._p.items():
            norm += p

        b_new._p = {index:p/norm for index,p in b_new._p.items() }
        return b_new

    def __rmul__(self,b1):
        return self.__mul__(b1)

    def copy(self):
        return copy.deepcopy(self)

    def update(self, class_label , weight=1):
        """
        class_label can be a name or an index

        We update the weight of the class by weight.
        """
        index = self._label_to_index(class_label)
        try:
            self._n[index] += weight
        except:
            print("Exception: Multinomial::update indexing error with self._n[index] += weight","self._n:",self._n,"index:",index)

        # TODO: move probabilities to log space
        total = sum([v for k,v in self._n.items()])
        self._p = { k : (n / total) for k,n in self._n.items()}

    def to_json_string(self):
        return(json.dumps(self.__dict__))

    def KL_Div(self,m1):
        """
        Calculates KL divergence between self and another multinomial
        distribution.
        """
        if not isinstance(m1,Multinomial):
            raise ValueError("Must multiply two Multinomial type objects. Type of argument is: {}".format(type(b1)))
        if self.size != m1.size:
            raise ValueError("Multinomial multiplication must be done on distributions of the same dimension. The two dimensions were {} and {}".format(self.size,m1.size))
        KL = 0.
        for i in range(self.size):
            p = self.get_p(i)
            q = m1.get_p(i)
            KL += p * math.log(p/q)
        return KL

    def set_prior_strength(self,prior_strength):
        """
        Set the prior strength, but keep the probabilities. Prior strength is 
        applied by taking the product of the prior strength with the probability
        for each class.
        """
        assert prior_strength > 0, "{} scale value must be positive".format(self.dist_type)
        self._n = {k:(prior_strength*self._p[k]) for k,v in self._n.items()}

    def get_cumulative_n(self):
        """
        returns the sum of all counts in _n. This represents the total 'weight'
        or confidience in the multinomial prior. If no scaling commands have
        been called, then this value represents exactly the number of data points
        that the distribution has been trained on.
        """
        return sum([v for k,v in self._n.items()])

def multinomial_from_json(json_str):
    obj = json.loads(json_str)
    m = Multinomial(1,1)
    m._n = {int(k):v for k,v in obj['_n'].items()} 
    m._p = {int(k):v for k,v in obj['_p'].items()}
    m.size = obj['size']
    m.labeled = obj['labeled']
    m.labels = obj['labels']
    m.dist_type = obj['dist_type']
    return m.copy()

if __name__ == '__main__':
    b1 = Multinomial(2)
    print("b1.__dict__",b1.__dict__)
    labels = {'funny':0, 'dumbshit':1}
    b1.relabel(labels)
    print (b1.get_p('funny'))
    print (b1.get_n('dumbshit'))
    print(b1)
    print("b1.__dict__",b1.__dict__)
    b2 = Multinomial(3)
    print(b2)
    print(b2.get_p(2))
    b2.update(0)
    b4 = Multinomial(size=3,prior_strength=1)
    print("b4:",b4)
    b4 = multinomial_from_json(b1.to_json_string())
    print("b4:",b4)
