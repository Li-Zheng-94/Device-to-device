import random

random.seed()  # 随机数种子


# 所有设备的公共接口类
class Interface(object):
    def __init__(self, i_id):
        self.__id = i_id

    def get_id(self):
        return self.__id


# 用户类
class User(Interface):
    def __init__(self, i_id):
        # 调用父类的构造函数
        Interface.__init__(self, i_id)

        self.__x_point = -1
        self.__y_point = -1
        self.__allocated_RB = []

    def set_location(self, x_point, y_point):
        self.__x_point = x_point
        self.__y_point = y_point

    def get_x_point(self):
        return self.__x_point

    def get_y_point(self):
        return self.__y_point

    def set_allocated_rb(self, rb_id):
        self.__allocated_RB.append(rb_id)

    def get_allocated_rb(self):
        return self.__allocated_RB


# 基站类
class BS(Interface):
    def __init__(self, i_id):
        # 调用父类的构造函数
        Interface.__init__(self, i_id)
        self.__power = 20  # 发射功率 dBm
        self.__x_point = 0
        self.__y_point = 0
        self.__allocated_RB = []

    def set_location(self, x_point, y_point):
        self.__x_point = x_point
        self.__y_point = y_point

    def get_x_point(self):
        return self.__x_point

    def get_y_point(self):
        return self.__y_point

    def get_power(self):
        return self.__power

    def set_allocated_rb(self, rb_id):
        self.__allocated_RB.append(rb_id)

    def get_allocated_rb(self):
        return self.__allocated_RB

