import random
import math

random.seed()  # 随机数种子


class Interface(object):
    def __init__(self, i_id):
        self.__id = i_id

    def get_id(self):
        return self.__id

