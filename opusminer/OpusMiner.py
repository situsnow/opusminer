import numpy as np
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.getcwd()))
# import the c++ shared library
if sys.version_info > (3, 0):
    # import the library compiled by boost-python3
    from opusminer.cpplib.boostpython3 import opus_miner
else:
    # import the library compiled by boost-python
    from opusminer.cpplib.boostpython2 import opus_miner


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
    # attribute_item_df = pd.DataFrame(columns=['attribute', 'value'])
    item_names = list()
    # save the size of value domain of each attribute, for non-market-basket data only
    attribute_value_size = list()
    # save the attribute index of each item
    value_attribute_mapping = list()
    # Save the transaction list according to item id
    transaction_ids = list()

    # size of each level of the searching space
    search_level_size = list()

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
                 print_closures=False,
                 market_basket=False,
                 ssi_filter=True,
                 correction_for_multicompare=True,
                 redundancy_tests=True
                 ):
        self.print_closures = print_closures
        self.filter = ssi_filter
        self.k = k
        self.search_by_lift = search_by_lift
        self.correction_for_multicompare = correction_for_multicompare
        self.redundancy_tests = redundancy_tests
        self.market_basket = market_basket
        self.input_data = input_data

    def fit(self):

        # total number of transactions in DB
        OpusMiner.noOfTransactions = len(self.input_data)

        # load data
        load_data(self, self.input_data)

        return opus_miner.opus_miner(self.print_closures, self.filter, self.k, self.search_by_lift,
                                     self.correction_for_multicompare, self.redundancy_tests, self.market_basket,
                                     self.noOfTransactions, self.noOfItems, len(self.attribute_value_size),
                                     self.search_level_size, self.item_names, self.transaction_ids)


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
            if all(str.isdigit(str(item)) for item in column_names):
                # data do not have column names
                field_name = 'field' + str(column + 1)
            else:
                field_name = column_names[column]

            # loop all unique items in current attribute
            for idx in range(len(unique_item)):
                # save item name
                OpusMiner.item_names.append(field_name + "=" + str(unique_item[idx]))
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
            OpusMiner.item_names.append(all_items[item])
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
    OpusMiner.search_level_size.append(math.log(search_size_df[0, len(search_size_df) - 1]))

    for j in range(2, no_of_attributes + 1):
        for k in range(j - 1, no_of_attributes):

            search_size_df[j - 1, k] = search_size_df[j - 2, k - 1] * att_value_domain[k] + \
                                       search_size_df[j - 1, k - 1]

            if k == no_of_attributes - 1:
                OpusMiner.search_level_size.append(math.log(search_size_df[j - 1, k]))

    OpusMiner.search_level_size = map(float, OpusMiner.search_level_size)

