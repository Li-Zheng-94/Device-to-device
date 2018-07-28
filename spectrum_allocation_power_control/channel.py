import math
import random


# 信道类
class Channel(object):
    def __init__(self, rx_id):
        self.__rx_id = rx_id
        self.__link_loss = {}  # 存储链路损耗的字典 键——发射机id值 值——链路损耗
        self.__id2distance = {}  # 存储距离的字典 键——发射机id值 值——距离

    def update_link_loss(self, tx_device, rx_device):
        distance = get_distance(tx_device.get_x_point(), tx_device.get_y_point(),
                                rx_device.get_x_point(), rx_device.get_y_point())
        self.__id2distance[tx_device.get_id()] = distance

        if rx_device.get_type() == 'BS':
            # 根据信道仿真标准计算路径损耗（套公式）
            link_loss = 128.1 + 37.6 * math.log10(distance/1000)
            shadow = random.normalvariate(0, 10)
            # shadow = 0
        else:
            # 根据信道仿真标准计算路径损耗（套公式）
            link_loss = 128.1 + 37.6 * math.log10(distance / 1000)
            shadow = random.normalvariate(0, 10)
            # shadow = 0
        self.__link_loss[tx_device.get_id()] = link_loss + shadow

    def get_rx_id(self):
        return self.__rx_id

    def get_link_loss(self, tx_id):
        return self.__link_loss[tx_id]

    def get_distance(self, tx_id):
        return self.__id2distance[tx_id]


def get_distance(x1, y1, x2, y2):
    return pow(pow((x1 - x2), 2) + pow((y1 - y2), 2), 0.5)
