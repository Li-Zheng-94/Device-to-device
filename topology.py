from channel import *
from resource_allocation import *
import random
import math
import numpy as np


# 单小区拓扑类
class SingleCell(object):
    def __init__(self, radius, cue_num, d2d_num, rb_num, up_or_down_link, d_tx2rx):
        self.__radius = radius  # 小区半径
        self.__cue_num = cue_num
        self.__d2d_num = d2d_num
        self.__rb_num = rb_num
        self.__up_or_down_link = up_or_down_link
        self.__d_tx2rx = d_tx2rx  # D2D发射机与接收机之间的最大距离

        self.__dict_id2device = {}  # id-设备对象登记表
        self.__dict_id2rx = {}  # id-接收机对象登记表
        self.__dict_id2tx = {}  # id-发射机对象登记表
        self.__dict_id2channel = {}  # 接收机id-信道对象登记表
        self.__observations = {}
        self.__observations_ = {}
        self.__dict_tx_id2sinr = {}
        self.__reward = 0
        self.__action = -1

    def initial(self):
        # 生成蜂窝用户对象
        for i_id in range(1, 1+self.__cue_num):
            cue = CUE(i_id, 'CUE')
            x, y = self.random_position()
            cue.set_location(x, y)
            self.__dict_id2device[i_id] = cue
            if self.__up_or_down_link == 'down':
                self.__dict_id2rx[i_id] = cue
            else:
                self.__dict_id2tx[i_id] = cue

        # 生成基站对象
        bs = BS(0, 'BS')  # 一个基站 id = 0
        self.__dict_id2device[0] = bs
        if self.__up_or_down_link == 'down':
            self.__dict_id2tx[0] = bs
        else:
            for tx_id in self.__dict_id2tx:
                bs.set_tx(tx_id)
            self.__dict_id2rx[0] = bs

        # 生成D2D对象
        for i_id in range(1+self.__cue_num, 1+self.__cue_num+self.__d2d_num):
            # D2D发射机对象
            d2d_tx = D2DTx(i_id, 'D2DTx')
            tx_x, tx_y = self.random_position()
            d2d_tx.set_location(tx_x, tx_y)
            d2d_tx.make_pair(i_id+self.__d2d_num)

            # D2D接收机对象
            d2d_rx = D2DRx(i_id+self.__d2d_num, 'D2DRx')
            rx_x, rx_y = self.d2d_rx_position(self.__d_tx2rx, tx_x, tx_y)
            d2d_rx.set_location(rx_x, rx_y)
            d2d_rx.make_pair(i_id)

            self.__dict_id2device[i_id] = d2d_tx
            self.__dict_id2tx[i_id] = d2d_tx
            self.__dict_id2device[i_id+self.__d2d_num] = d2d_rx
            self.__dict_id2rx[i_id+self.__d2d_num] = d2d_rx

        # 生成信道 一个接收机对应一个信道对象
        for rx_id in self.__dict_id2rx:  # 遍历所有的接收机
            temp_channel = Channel(rx_id)

            for tx_id in self.__dict_id2tx:  # 遍历所有的发射机
                temp_channel.update_link_loss(self.__dict_id2tx[tx_id], self.__dict_id2rx[rx_id])

                self.__dict_id2channel[temp_channel.get_rx_id()] = temp_channel

    # 在单小区范围内随机生成位置
    def random_position(self):
        theta = random.random() * 2 * math.pi
        r = random.uniform(0, self.__radius)
        x = r * math.sin(theta)
        y = r * math.cos(theta)
        return x, y

    # 根据发射机的位置，在半径 r 范围内随机生成接收机位置
    def d2d_rx_position(self, r, tx_x, tx_y):
        x, y = self.__radius, self.__radius
        while x**2 + y**2 > self.__radius**2:
            theta = random.random() * 2 * math.pi
            r = random.uniform(0, r)
            x = r * math.sin(theta) + tx_x
            y = r * math.cos(theta) + tx_y
        return x, y

    # 运行仿真流程
    def work(self, slot, RL):
        # 随机分配信道
        if slot == 0:
            random_allocation(self.__dict_id2tx, self.__dict_id2rx, self.__rb_num)
        else:
            random_allocation(self.__dict_id2tx, self.__dict_id2rx, self.__rb_num)
            for tx_id in self.__dict_id2tx:
                tx_id = 11  # 测试 只训练 D2DTx 11
                temp_tx = self.__dict_id2tx[tx_id]
                if temp_tx.get_type() == 'D2DTx':
                    if slot > 1:
                        observation = np.array(self.__observations[11])
                        observation_ = temp_tx.get_observation()
                        observation_ = np.array(observation_)
                        # 存储记忆
                        RL.store_transition(observation, self.__action, self.__reward, observation_)

                    # 当回合数大于200后，每2回合学习1次（先积累一些记忆再开始学习）
                    if (slot > 20) and (slot % 5 == 0):
                        RL.learn()

                    observation = temp_tx.get_observation()
                    self.__observations[tx_id] = observation
                    observation = np.array(observation)
                    # 根据状态选择行为
                    action = RL.choose_action(observation)
                    self.__action = action
                    rb_id = action
                    temp_tx.set_allocated_rb(rb_id)

        # 计算SINR
        for rx_id in self.__dict_id2rx:  # 遍历所有的接收机
            inter = self.__dict_id2rx[rx_id].comp_sinr(self.__dict_id2tx, self.__dict_id2channel)
            sinr = self.__dict_id2rx[rx_id].get_sinr()
            if type(sinr) == float:  # D2D
                tx_id = self.__dict_id2rx[rx_id].get_tx_id()
                self.__dict_tx_id2sinr[tx_id] = sinr
                print('D2D接收机ID:' + str(rx_id) + ' SINR:' + str(sinr))
                # 统计previous类数据
                temp_rx = self.__dict_id2rx[rx_id]
                tx_id = temp_rx.get_tx_id()
                temp_tx = self.__dict_id2tx[tx_id]
                temp_tx.previous_rb = temp_tx.get_allocated_rb()[0]
                temp_tx.previous_inter = inter
                neighbors = self.get_neighbors(temp_rx, 3)
                temp_tx.previous_neighbor_1_rb = self.__dict_id2tx[neighbors[0]].get_allocated_rb()[0]
                temp_tx.previous_neighbor_2_rb = self.__dict_id2tx[neighbors[1]].get_allocated_rb()[0]
                temp_tx.previous_neighbor_3_rb = self.__dict_id2tx[neighbors[2]].get_allocated_rb()[0]
            else:  # CUE
                for tx_id in sinr:
                    print('基站对应的发射机ID:' + str(tx_id) + ' SINR:' + str(sinr[tx_id]))

        if slot != 0:
            if self.__dict_tx_id2sinr[11] > 30:
                self.__reward = 1
            else:
                self.__reward = -1

    # 更新用户位置
    def update(self):
        for tx_id in self.__dict_id2tx:
            temp_tx = self.__dict_id2tx[tx_id]
            if temp_tx.get_type() == 'D2DTx':
                rx_id = temp_tx.get_rx_id()
                d2d_channel = self.__dict_id2channel[rx_id]
                tx2bs_channel = self.__dict_id2channel[0]
                temp_tx.d2d_csi = d2d_channel.get_link_loss(tx_id)
                temp_tx.tx2bs_csi = tx2bs_channel.get_link_loss(tx_id)

    def get_neighbors(self, rx, num):
        neighbors = []
        tx_id2distance = {}
        for tx_id in self.__dict_id2tx:
            if rx.get_tx_id() != tx_id:
                temp_tx = self.__dict_id2tx[tx_id]
                distance = get_distance(temp_tx.get_x_point(), temp_tx.get_y_point(),
                                        rx.get_x_point(), rx.get_y_point())
                tx_id2distance[tx_id] = distance
        list_tx_id2distance = sorted(tx_id2distance.items(), key=lambda item: item[1])
        for i in range(num):
            neighbors.append(list_tx_id2distance[i][0])
        return neighbors
