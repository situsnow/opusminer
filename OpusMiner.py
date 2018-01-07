import pandas as pd
import numpy as np
import sys

import copy

from ItemsetRec import ItemsetRec
from ItemQClass import ItemQClass
from Associations import Associations, Itemset


class OpusMiner:
    """Open source implementation of the OPUS Miner algorithm which applies OPUS search for
    Filtered Top-k Association Discovery of Self-Sufficient Itemsets
    Copyright (C) 2012 Geoffrey I Webb

    This program is free software: you can redistribute it and/or modify it under the terms of the
    GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
    """

    # ################## Global variables #############################
    # total number of transactions in DB
    noOfTransactions = 0
    # total number of unique items in DB
    noOfItems = 0
    # attribute-value-item data frame
    attribute_item_df = pd.DataFrame(columns=['attribute', 'value'])
    # save the size of value domain of each attribute, for non-market-basket data only
    attribute_value_size = list()
    # save the attribute index of each item
    value_attribute_mapping = list()
    # Save the transaction list according to item id
    transaction_ids = list()
    # non-redundant productive itemsets
    global_itemsets = list()

    # for each itemset explored for which supersets might be in the best k, keep the count
    # the dictionary with key as the itemset, value as the transaction cover count
    TIDCount = dict()

    # the list to save alpha in different level of search space
    alpha = list()

    # the logarithm of factorial
    lf = list()

    # size of each level of the searching space
    search_level_size = list()

    # the minimum leverage of an itemset in the top-k so far
    # any itemset whose leverage does not exceed this value cannot enter the top-k
    if sys.maxsize > 2 ** 32:
        # for 64-bit pc:
        minValue = np.finfo("float64").min
    else:
        minValue = np.finfo("float32").min

    """
    :param
    @print_closure - Each output itemset is followed by its closure
    @filter - Suppress filtering out itemsets that are not independently productive.
    @k - Set k to the integer value <i>.  By default it is 100
    @search_by_lift - Set the measure of interest to lift.  By default it is leverage.
    @correction_for_multicompare - True: perform multiple layer correction in Fisher Exact Test; False: otherwise
    @redundancy_test - Allow redundant itemsets.
    @input_date - the data frame that imported from the input file
        with column name as the attribute name if not market-basket data
    """
    def __init__(self,
                 k,
                 search_by_lift,
                 input_data,
                 # output_file,
                 print_closures=False,
                 market_basket=False,
                 ssi_filter=True,
                 correction_for_multicompare=True,
                 redundancy_tests=True
                 ):
        self.print_closures = print_closures
        self.filter = ssi_filter,
        self.k = k
        self.search_by_lift = search_by_lift
        self.correction_for_multicompare = correction_for_multicompare
        self.redundancy_tests = redundancy_tests
        self.market_basket = market_basket
        self.input_data = input_data
        # self.output_file = output_file

    def fit(self):

        # total number of transactions in DB
        OpusMiner.noOfTransactions = len(self.input_data)

        # load data
        load_data(self, self.input_data)

        # find_itemsets
        find_itemsets(self)

        # end time of searching for items
        # find_end_t = datetime.datetime.now()

        # Sort the global itemsets (list of itemset (list))
        itemset = []
        while OpusMiner.global_itemsets:
            current_itemset_record = OpusMiner.global_itemsets[0]
            itemset.append(current_itemset_record)
            OpusMiner.global_itemsets.remove(current_itemset_record)

        # filter_itemsets
        if self.filter:
            filter_itemsets(self, itemset)

        # save the rules into the Opus object and return
        return print_itemsets(self, itemset)


def load_data(opus_miner_obj, input_data):
    if not opus_miner_obj.market_basket:
        item = 0

        # get column name
        column_names = input_data.columns.values

        # each variable should belong to one single attribute
        for column in range(len(column_names)):

            unique_item = list(set(input_data[column_names[column]]))

            # save the value domain size of current attribute
            OpusMiner.attribute_value_size.append(len(unique_item))

            # get field name
            if all(np.issubdtype(type(item), int) for item in column_names):
                # data do not have column names
                field_name = 'field' + str(column + 1)
            else:
                field_name = column_names[column]

            # loop all unique items in current attribute
            for idx in range(len(unique_item)):
                # save all item into attribute_item_df
                if OpusMiner.attribute_item_df.empty:
                    first_record = pd.DataFrame({'attribute': field_name, 'value': unique_item[idx]},
                                                index=[0])
                    OpusMiner.attribute_item_df = OpusMiner.attribute_item_df.append(first_record)
                else:
                    OpusMiner.attribute_item_df = OpusMiner.attribute_item_df.append({'attribute': field_name,
                                                                                      'value': unique_item[idx]},
                                                                                     ignore_index=True)
                # save the item and attribute mapping
                OpusMiner.value_attribute_mapping.append(column)

                # save all transactions has current item
                cover = [i for i, value in enumerate(input_data[column_names[column]]) if value == unique_item[idx]]
                OpusMiner.transaction_ids.append(cover)
                item += 1
        OpusMiner.noOfItems = item

        calculate_layer_size()

    else:
        # each variable in tuple should be treated in same level
        all_items = [item for item in np.unique(input_data.values.ravel()) if item != '']
        OpusMiner.noOfItems = len(all_items)
        for item in range(len(all_items)):
            OpusMiner.attribute_item_df.append({'attribute': '', 'value': all_items[item]}, ignore_index=True)
            indices = input_data.apply(lambda row: row.astype(str).str.contains(all_items[item]).any(), axis=1)
            cover = [idx for idx, value in enumerate(indices) if value]
            OpusMiner.transaction_ids.append(cover)


# In non-market-basket data, when examining an itemset, no items from same attribute should be presented together
def calculate_layer_size():

    # get the the unique list of attributes
    no_of_attributes = len(OpusMiner.attribute_value_size)

    # Available for non-market-basket data only
    # the matrix to save the search space of different layers
    # in j-th row where j is the size of itemset (layers of search space)
    # in k-th column where k is the index of attributes
    # df[j,k] is the total number of itemsets that exist in the search space
    # formula is from - Webb, G. I. (2007). Discovering significant patterns. Machine Learning, 68(1), 1-33.
    search_size_df = np.zeros(shape=(no_of_attributes, no_of_attributes))

    att_value_domain = []

    for i in range(no_of_attributes):

        # attribute_name = unique_att_list[i]

        # for each attribute, get value domain size
        value_domain_size = OpusMiner.attribute_value_size[i]

        if i == 0:
            search_size_df[0, 0] = value_domain_size
        else:
            search_size_df[0, i] = search_size_df[0, i - 1] + value_domain_size

        att_value_domain.append(value_domain_size)

    # save the actual search level size
    OpusMiner.search_level_size.append(0)

    # save the search level size in first level
    OpusMiner.search_level_size.append(search_size_df[0, len(search_size_df) - 1])

    for j in range(2, no_of_attributes + 1):
        for k in range(j - 1, no_of_attributes):

            search_size_df[j - 1, k] = search_size_df[j - 2, k - 1] * att_value_domain[k] + \
                                       search_size_df[j - 1, k - 1]

            if k == no_of_attributes - 1:
                OpusMiner.search_level_size.append(search_size_df[j - 1, k])


# ################## functions from find_itemsets.cpp ###################
def find_itemsets(opus_miner_obj):

    # a queue of items, to be sorted on an upper bound on value
    q_class = ItemQClass()

    # initialize q - the queue of items ordered on an upper bound on value
    for item in range(0, OpusMiner.noOfItems):
        # print(item)
        cover_size = len(OpusMiner.transaction_ids[item])
        sup = count_to_support(cover_size)

        upper_bound_value = 0.0
        if opus_miner_obj.search_by_lift:
            upper_bound_value = 1.0 / sup
        else:
            upper_bound_value = sup - sup * sup

        # make sure that the support is high enough for it to be possible to create a significant itemset
        if fisher(cover_size, cover_size, cover_size) <= get_alpha(opus_miner_obj, 2):

            # it is faster to sort the q once it is full rather than doing an insertion sort
            q_class.append(upper_bound_value, item)

    # this is the queue of items that will be available for the item currently being explored
    new_q = ItemQClass()

    if len(q_class) > 0:
        q_class = q_class.sort()
        """the first item will have no previous items with which to be paired so is simply 
        added to the queue of available items"""
        new_q.insert(q_class[0].ubVal, q_class[0].item)

    # remember the current minValue, and output an update if it improves in this iteration of the loop
    prevMinVal = OpusMiner.minValue

    # print("total amount: " + str(len(q_class)))
    # we are stepping through all associations of i with j<i, so the first value of i that will have effect is 1
    for i in range(1, len(q_class)):

        # print(i)

        if q_class[i].ubVal <= OpusMiner.minValue:
            break
        item = q_class[i].item

        itemset = ItemsetRec()
        itemset.append(item)

        opus(opus_miner_obj, itemset, OpusMiner.transaction_ids[item], new_q, len(OpusMiner.transaction_ids[item]), item)

        new_q.append(q_class[i].ubVal, item)

        if prevMinVal < OpusMiner.minValue:
            # print("<%f" & minValue)
            prevMinVal = OpusMiner.minValue
        # else:
            # print(".")
        # print("\n")


# perform OPUS search for specialisations of is (which covers cover) using the candidates in queue q
# maxItemSup is the maximum of the supports of all individual items in is
def opus(opus_miner_obj, itemset, cover, q_class, max_item_count, new_added_item):

    parent_sup = count_to_support(len(cover))
    depth = len(itemset) + 1

    new_queue = ItemQClass()

    for i in range(len(q_class)):
        item = q_class[i].item

        # check if new_added_item and item are from same attribute, available to non-market-basket data only
        if (not opus_miner_obj.market_basket) and OpusMiner.value_attribute_mapping[new_added_item] == \
                OpusMiner.value_attribute_mapping[item]:
            continue

        # determine the number of TIDs that the new itemset covers
        new_cover = list(set(cover) & set(OpusMiner.transaction_ids[item]))
        count = len(new_cover)

        new_max_item_count = max(max_item_count, len(OpusMiner.transaction_ids[item]))
        new_sup = count_to_support(count)

        # this is a lower bound on the p value that may be obtained for this itemset or any superset
        lb_p = fisher(count, new_max_item_count, count)

        # calculate an upper bound on the value that can be obtained by this itemset or any superset
        if opus_miner_obj.search_by_lift:
            if count == 0:
                upper_bound_value = 0.0
            else:
                upper_bound_value = 1.0 / count_to_support(max_item_count)
        else:
            upper_bound_value = new_sup - new_sup * count_to_support(max_item_count)

        # performing OPUS pruning - if this test fails, the item will not be included in any superset of itemset
        if lb_p <= get_alpha(opus_miner_obj, depth) and upper_bound_value > OpusMiner.minValue:

            # only continue if there is any possibility of this itemset or its supersets entering the list of best
            # itemsets
            itemset.append(item)

            [redundant, apriori] = check_immediate_subsets(opus_miner_obj, itemset, count)

            if not apriori:
                [val, p, flag] = check_subsets(opus_miner_obj, item, itemset, count, new_sup, len(cover), parent_sup,
                                               get_alpha(opus_miner_obj, depth))
                if flag:
                    itemset.count = count
                    itemset.value = val
                    itemset.p = p
                    insert_itemset(opus_miner_obj, itemset)

                # performing OPUS pruning - if this test fails, the item will not be included in any superset of itemset
                # the redundancy test here means the superset of current itemset will be redundant
                if not redundant:
                    # sorted before inserting into dictionary
                    # tmp_is = copy.deepcopy(itemset)
                    tmp_is = itemset * 1
                    tmp_is = sorted(tmp_is)

                    OpusMiner.TIDCount[repr(tmp_is)] = count
                    if len(new_queue) != 0:

                        # there are only more nodes to expand if there is a queue of items to consider expanding it with
                        opus(opus_miner_obj, itemset, new_cover, new_queue, new_max_item_count, item)

                    new_queue.insert(upper_bound_value, item)

            itemset.remove(item)


def check_immediate_subsets(opus_miner_obj, itemset, itemset_cnt):
    current_subset = itemset.copy()

    # the redundancy test (second property of principle of self-sufficient itemset) flag of current itemset
    redundant = False
    # the flag of apriori here is nothing related to Algorithm Apriori
    # but the definition of anti-monotonicity property of an items and its subset.
    # If any immediate subsets of current itemset cannot pass the test, current itemset should not be considered.
    apriori = False

    for it in range(len(itemset)):
        current_subset.remove(current_subset[it])

        [flag, subset_cnt] = get_tid_count(current_subset)
        if not flag:
            redundant = True
            apriori = True
            return [redundant, apriori]

        if opus_miner_obj.redundancy_tests and subset_cnt == itemset_cnt:
            redundant = True

        current_subset.insert(it, itemset[it])

    return [redundant, apriori]


# calculates leverage, p, whether is is redundant and whether it is possible to determine that all supersets of itemset
# will be redundant return true iff is is not redundant, val > minValue and p <= alpha
def check_subsets(opus_miner_obj, item, itemset, cnt, new_sup, parent_cnt, parent_sup, alpha):

    # do test for new item against the rest
    item_cnt = len(OpusMiner.transaction_ids[item])

    # do test for so_far against remaining
    if opus_miner_obj.search_by_lift:
        val = new_sup / (parent_sup * item_support(item))
    else:
        val = new_sup - parent_sup * item_support(item)

    if val <= OpusMiner.minValue:
        return [0.0, 0.0, False]

    p = fisher(cnt, item_cnt, parent_cnt)

    if p > alpha:
        return [0.0, 0.0, False]

    if len(itemset) > 2:
        so_far = list()
        # remaining = copy.deepcopy(itemset)
        remaining = itemset * 1

        so_far.append(item)
        remaining.remove(item)

        for it in range(len(itemset)):
            if itemset[it] != item:
                so_far.append(itemset[it])
                remaining.remove(itemset[it])

                [val, p, flag] = check_subsets_x(opus_miner_obj, so_far, remaining, itemset[it], cnt, new_sup,
                                                 val, p, alpha)
                if not flag:
                    return [val, p, False]

                so_far.remove(itemset[it])
                remaining.append(itemset[it])

    return [val, p, p <= alpha and val > OpusMiner.minValue]


# calculates leverage, p, whether the itemset is is redundant and whether it is possible to determine that all
# supersets of is will be redundant return true iff is is not redundant, val > minValue and p <= alpha
def check_subsets_x(opus_miner_obj, so_far, remaining, limit, cnt, new_sup, val, p, alpha):

    [so_far_flag, so_far_cnt] = get_tid_count(so_far)
    [reminaing_flag, remaining_cnt] = get_tid_count(remaining)

    if (not so_far_flag) or (not reminaing_flag):
        return [val, p, False]

    # do test for sofar against remaining
    if opus_miner_obj.search_by_lift:
        this_val = new_sup / (count_to_support(remaining_cnt) * count_to_support(so_far_cnt))
    else:
        this_val = new_sup - count_to_support(remaining_cnt) * count_to_support(so_far_cnt)

    if this_val > val:
        val = this_val

        if this_val <= OpusMiner.minValue:
            return [val, p, False]

    this_p = fisher(cnt, so_far_cnt, remaining_cnt)

    if this_p > p:
        p = this_p
        if p > alpha:
            return [val, p, False]

    if len(remaining) > 1:
        # new_remaining = copy.deepcopy(remaining)
        new_remaining = remaining * 1

        for it in range(len(remaining)):
            if remaining[it] >= limit:
                break
            so_far.append(remaining[it])
            new_remaining.remove(remaining[it])

            [val, p, flag] = check_subsets_x(opus_miner_obj, so_far, new_remaining, remaining[it], cnt, new_sup,
                                             val, p, alpha)

            if not flag:
                return [val, p, False]

            so_far.remove(remaining[it])
            new_remaining.append(remaining[it])

    return [val, p, (p <= alpha) and val > OpusMiner.minValue]


# insert itemset into the collection of k best itemsets
def insert_itemset(opus_miner_obj, itemset):

    if OpusMiner.global_itemsets and len(OpusMiner.global_itemsets) >= opus_miner_obj.k:
        # remove the itemset which has smallest value
        OpusMiner.global_itemsets.remove(OpusMiner.global_itemsets[0])

    tmp_copy = copy.deepcopy(itemset)
    # tmp_copy = itemset.copy()

    OpusMiner.global_itemsets.append(tmp_copy)

    # sort the g_itemsets again with ascending order
    OpusMiner.global_itemsets = sorted(OpusMiner.global_itemsets, key=lambda itemset_rec: itemset_rec.value)

    if len(OpusMiner.global_itemsets) == opus_miner_obj.k:
        new_min = OpusMiner.global_itemsets[0].value

        if new_min > OpusMiner.minValue:
            OpusMiner.minValue = new_min


# access function for TIDCount
def get_tid_count(itemset):
    if len(itemset) == 1:
        count = len(OpusMiner.transaction_ids[itemset[0]])
        return [True, count]
    else:
        # tmp_is = copy.deepcopy(itemset)
        tmp_is = itemset * 1
        tmp_is = sorted(tmp_is)
        if repr(tmp_is) in OpusMiner.TIDCount:
            count = OpusMiner.TIDCount.get(repr(tmp_is))
            return [True, count]
        else:
            return [False, 0]


# ################## functions from globals.cpp ###################
def expand_alpha(opus_miner_obj, depth):
    if not OpusMiner.alpha:
        # alpha[0] and [1] are not used.
        OpusMiner.alpha.append(1.0)
        OpusMiner.alpha.append(1.0)
        if depth <= 1:
            return

    if opus_miner_obj.market_basket:
        if depth > OpusMiner.noOfItems:
            OpusMiner.alpha.append(0.0)
        elif depth == OpusMiner.noOfItems:
            # at deepest level so might as well use as much of the rest of the probability mass as possible
            OpusMiner.alpha.append(OpusMiner.alpha[depth - 1])
        else:
            for i in range(len(OpusMiner.alpha), depth + 1):
                OpusMiner.alpha.append(
                    min(
                        (
                            np.power(0.5, depth - 1) /
                            np.exp(
                                log_combine(OpusMiner.noOfItems, depth)
                            )
                        ) * 0.5,
                        OpusMiner.alpha[depth - 1]
                    )
                )

    else:
        if depth > len(OpusMiner.attribute_value_size):
            OpusMiner.alpha.append(0.0)
        elif depth == len(OpusMiner.attribute_value_size):
            # at deepest level so might as well use as much of the rest of the probability mass as possible
            OpusMiner.alpha.append(OpusMiner.alpha[depth - 1])
        else:
            for i in range(len(OpusMiner.alpha), depth + 1):

                # According to the multiple correction formula, change the size(L) according to actual search space
                OpusMiner.alpha.append(
                    min(
                        (
                            np.power(0.5, depth - 1) / OpusMiner.search_level_size[depth]
                        ) * 0.05,
                        OpusMiner.alpha[depth - 1]
                    )
                )


def get_alpha(opus_miner_obj, depth):
    if not opus_miner_obj.correction_for_multicompare:
        return 0.05

    existing_length_alpha = 0
    if OpusMiner.alpha:
        existing_length_alpha = len(OpusMiner.alpha)

    if depth >= existing_length_alpha:
        expand_alpha(opus_miner_obj, depth)

    return OpusMiner.alpha[depth]


# ################## functions from fisher.cpp ##################
# return the p value for a one tailed fisher exact test for the probability of obtaining d or more in a
# contingency table where the marginal frequencies are invariant
def fisher_test(a, b, c, d):
    p = 0

    # will loop until b or c is 0 - as the values are interchangeable, make c the lesser value
    # and test only for when it reaches 0
    if b < c:
        t = b
        b = c
        c = t

    # use log factorial to scale down the Fisher Exact Test result in case large number
    invariant = -log_factorial(a + b + c + d) + log_factorial(a + b) + log_factorial(c + d) +\
                log_factorial(a + c) + log_factorial(b + d)

    while c >= 0:
        # calculate the p value by raising the power
        p += np.exp(invariant - log_factorial(a) - log_factorial(b) - log_factorial(c) - log_factorial(d))
        a += 1
        b -= 1
        c -= 1
        d += 1

    return p


def log_combine(n, k):
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)


# return the log of the factorial of n
def log_factorial(n):
    for i in range(len(OpusMiner.lf), n+1):
        if i == 0:
            OpusMiner.lf.append(0)
        else:
            OpusMiner.lf.append(OpusMiner.lf[i-1]+np.log(i))
    return OpusMiner.lf[n]


# ############################# functions from utils.cpp #############################

# true if and only if s1 is a subset of s2
def subset(s1, s2):
    if set(s1) < set(s2):
        return True
    else:
        return False


def get_tids(itemset):
    transactions = list()

    if len(itemset) == 1:
        transactions = OpusMiner.transaction_ids[itemset[0]]
    else:
        transactions = list(set(OpusMiner.transaction_ids[itemset[0]]) & set(OpusMiner.transaction_ids[itemset[1]]))

        for index in range(2, len(itemset)):
            transactions = list(set(transactions) & set(OpusMiner.transaction_ids[itemset[index]]))

    return transactions


def count_to_support(count):
    return count/OpusMiner.noOfTransactions


def item_support(item):
    return count_to_support(len(OpusMiner.transaction_ids[item]))


# return the result of a Fisher exact test for an itemset i with support count count
# relative to support counts count1 and count2 for two subsets s1 and s2 that form a partition of i
def fisher(count, count1, count2):
    return fisher_test(OpusMiner.noOfTransactions - count1 - count2 + count, count1 - count, count2 - count, count)


# ############################# functions from print_itemsets.cpp #############################
def print_itemset(opus_miner_obj, itemset):

    itemset_str = ""
    for i in range(len(itemset)):

        if i != 0:
            itemset_str += " & "

        if opus_miner_obj.market_basket:
            # get the item name according to the item index
            itemset_str += OpusMiner.attribute_item_df.iloc[itemset[i], 0]
        else:
            # get the field and value according to the item index
            itemset_str += OpusMiner.attribute_item_df.iloc[itemset[i], 0] + " = " + \
                           OpusMiner.attribute_item_df.iloc[itemset[i], 1]

    return itemset_str


def print_itemset_record(opus_miner_obj, itemsetRec):

    top_k_itemset = Itemset()

    top_k_itemset.itemset_str = print_itemset(opus_miner_obj, itemsetRec)

    top_k_itemset.rule_size = len(itemsetRec)

    top_k_itemset.value = getattr(itemsetRec, "value")

    top_k_itemset.p_value = getattr(itemsetRec, "p")

    if opus_miner_obj.print_closures:

        closure = find_closure(itemsetRec)

        if len(closure) > len(itemsetRec):

            top_k_itemset.closure_itemset_str = print_itemset(opus_miner_obj, closure)

    return top_k_itemset


def print_itemsets(opus_miner_obj, itemset):

    opus_obj = Associations()

    opus_obj.alpha_list = OpusMiner.alpha

    # sort with descending order of itemset value
    itemset = sorted(itemset, key=lambda item_set: item_set.value, reverse=True)

    failed_count = 0

    for index in range(0, len(itemset)):
        if not getattr(itemset[index], 'self_sufficient'):
            failed_count += 1
        else:
            ssi_obj = print_itemset_record(opus_miner_obj, itemset[index])
            opus_obj.self_sufficient_itemsets.append(ssi_obj)

    if failed_count > 0:
        for index in range(0, len(itemset)):
            if not getattr(itemset[index], 'self_sufficient'):
                non_ssi_obj = print_itemset_record(opus_miner_obj, itemset[index])
                opus_obj.non_self_sufficient_itemsets.append(non_ssi_obj)

    return opus_obj


# ############################# functions from find_closure.cpp #############################
def find_closure(itemset):

    closure = itemset * 1
    this_tids = get_tids(itemset)

    for item in range(0, OpusMiner.noOfItems):
        if len(OpusMiner.transaction_ids[item]) >= len(this_tids) and (item in itemset) \
                and (len(set(this_tids) & set(OpusMiner.transaction_ids[item]))) == len(this_tids):
            closure.append(item)

    return closure


# ############################# functions from filter_itemsets.cpp #############################
# check whether itemsets can be explained by their supersets
def filter_itemsets(opus_miner_obj, itemset):
    if itemset:
        """Sort the itemsets so that the largest are first.
        This way we only need to look at the itemsets before one that we are processing to find superset.
        Also, we will determine whether a superset is self sufficient 
        before trying to determine whether its subsets are"""
        itemset = sorted(itemset, key=len, reverse=True)

        for subset_it in range(1, len(itemset)):
            # get the TIDs that are covered by the current itemset's supersets
            supset_tids = []  # the tids covered by the supitems

            for supset_it in range(len(itemset)):

                if supset_it == subset_it:
                    break

                if itemset[supset_it].self_sufficient:

                    if subset(itemset[subset_it], itemset[supset_it]):

                        # the additional items n the supersets of the current itemset
                        sup_items = list(set(itemset[supset_it]) - set(itemset[subset_it]))

                        if sup_items:
                            this_supset_tids = get_tids(sup_items)  # the tids covered by the supitems

                            if supset_tids:
                                supset_tids = this_supset_tids
                            else:
                                supset_tids = list(set(supset_tids) | set(this_supset_tids))

            if len(supset_tids) > 0 and not check_self_sufficient(opus_miner_obj, itemset[subset_it], supset_tids):
                # only call checkSS if one or more supersets were found (and hence TIDs retrieved
                itemset[subset_it].self_sufficient = False


# check whether itemset is is self sufficient given that it has supersets that cover the TIDs in supsettids
def check_self_sufficient(opus_miner_obj, itemset, supsettids):
    result = True

    # find for each item in is the TIDs that it covers that are not in supsettids
    unique_tids = []
    for it in range(len(itemset)):

        set_difference = list(set(OpusMiner.transaction_ids[itemset[it]]) - set(supsettids))
        if len(set_difference) == 0:
            # there cannot be a significant association from adding this tidset
            result = False
            break

        unique_tids.append(set_difference)

    if result:
        """set up a process that will check whether unique_cov.size() is significantly greater than can be predicted 
        by assuming independence between any partition of itemset"""
        unique_cov = list(unique_tids[0])  # this is the TIDs covered by is that are not in supsettids

        for i in range(len(itemset)):
            unique_cov = list(set(unique_cov) & set(unique_tids[i]))

        # this is the cover of the items committed to the right - initialise it to the last unique TID
        tidsright = list(unique_tids[len(unique_tids) - 1])

        # start with the last item committed to the right, then successively commit eeach item first to the left
        # then to the right
        for i in range(len(unique_tids) - 2, 0, -1):
            result = check_self_sufficient2(unique_tids, i, unique_tids[i], tidsright, OpusMiner.noOfTransactions - len(supsettids),
                                            len(unique_cov), get_alpha(opus_miner_obj, len(itemset)))

            if not result:
                return False

            if i > 0:
                tidsright = list(set(tidsright) & set(unique_tids[i]))

    return result


# check all combinations of intersections of partitions of tids_avail moved to either tids_left or tids_right
def check_self_sufficient2(unique_tids, no, tids_left, tids_right, tids_available, count, alpha):
    if no == 0:
        if fisher_test(tids_available - len(tids_left) - len(tids_right) + count,
                       len(tids_left) - count, len(tids_right) - count, count) > alpha:
            return False
        else:
            return True

    # first try with the tidset committed to the left then try with it committed to the right
    new_tids = list(set(unique_tids[no - 1]) & set(tids_right))

    if not check_self_sufficient2(unique_tids, no-1, new_tids, tids_right, tids_available, count, alpha):
        return False

    new_tids = list(set(unique_tids[no - 1]) & set(tids_right))

    if not check_self_sufficient2(unique_tids, no-1, tids_left, new_tids, tids_available, count, alpha):
        return False

    return True
