# ############################# Class Itemset/ItemsetRec #############################
class ItemsetRec(list):
    def __init__(self,
                 count=0,
                 value=0.0,
                 p=1.0,
                 self_sufficient=True):
        super().__init__(self)
        self.count = count
        self.value = value
        self.p = p
        self.self_sufficient = self_sufficient

    # override the greater > operator
    @staticmethod
    def __gt__(pi1, pi2):
        return pi1.value > pi2.value

    @staticmethod
    def sizegt(self, i2):
        if len(self) < len(i2):
            return -1
        elif len(self) > len(i2):
            return 1
        else:
            return 0

    """def __deepcopy__(self, memodict={}):
        copy_object = self * 1
        copy_object.count = self.count
        copy_object.value = self.value
        copy_object.p = self.p
        copy_object.self_sufficient = self.self_sufficient"""

