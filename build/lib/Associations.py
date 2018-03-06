

class Associations:

    # the first index will be the alpha value in layer 1
    alpha_list = []

    # list of SelfSufficientItemset objects that pass the self-sufficiency principle test
    self_sufficient_itemsets = []

    # list of SelfSufficientItemset objects that DO NOT pass the self-sufficiency principle test
    non_self_sufficient_itemsets = []


class Itemset:

    itemset_str = ""

    closure_itemset_str = ""

    # cover = 0

    # leverage = 0.0

    # lift = 0.0

    rule_size = 0

    value = 0.0

    p_value = 0.0

    # ant_sup = 0.0

    # confidence
    # strength = 0.0
