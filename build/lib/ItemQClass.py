# ############################# Class ItemQElem/ItemQClass #############################
class ItemQElem:
    def __init__(self, item, ubVal=0.0):
        self.item = item
        self.ubVal = ubVal

    @staticmethod
    def iqeGreater(iqe):
        return iqe.ubVal


class ItemQClass(list):
    def __init__(self):
        list.__init__(self)

    def append(self, ubVal, item):
        new_itemQElem = ItemQElem(item, ubVal)
        super(ItemQClass, self).append(new_itemQElem)

    def sort(self):
        return sorted(self, key=ItemQElem.iqeGreater, reverse=True)

    def insert(self, ubVal, item):

        new_itemQElem = ItemQElem(item, ubVal)
        super(ItemQClass, self).append(new_itemQElem)

        # sort current ItemQClass to ensure item sorted with upper value in descending order
        self.sort()
