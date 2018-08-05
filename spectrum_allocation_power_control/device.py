import random
import math
import numpy as np
from spectrum_allocation_power_control.q_learning import QLearningTable

random.seed()  # 随机数种子


# 所有设备的公共接口类
class Interface(object):
    def __init__(self, i_id, i_type):
        self.__id = i_id
        self.__type = i_type

    def get_id(self):
        return self.__id

    def get_type(self):
        return self.__type


# 基站类
class BS(Interface):
    def __init__(self, i_id, i_type):
        # 调用父类的构造函数
        Interface.__init__(self, i_id, i_type)
        self.__power = 40  # 发射功率 dBm
        self.__x_point = 0
        self.__y_point = 0
        self.__allocated_rb = []
        self.__txs_id = []  # 上行链路 蜂窝链路发射机登记表
        self.__rxs_id = []  # 下行链路 蜂窝链路接收机登记表
        self.__tx_id2sinr = {}  # 发射机ID——接收SINR登记表

    def update_location(self):
        pass

    def set_location(self, x_point, y_point):
        self.__x_point = x_point
        self.__y_point = y_point

    def get_x_point(self):
        return self.__x_point

    def get_y_point(self):
        return self.__y_point

    def get_power(self):
        return self.__power

    def clear_allocated_rb(self):
        self.__allocated_rb = []

    def set_allocated_rb(self, rb_id):
        self.__allocated_rb.append(rb_id)

    def get_allocated_rb(self):
        return self.__allocated_rb

    def set_rx(self, rx_id):
        self.__rxs_id.append(rx_id)

    def set_tx(self, tx_id):
        self.__txs_id.append(tx_id)

    def get_tx_id(self):
        return self.__txs_id

    def comp_sinr(self, dict_id2tx, dict_id2channel):  # 计算接收 SINR
        if len(self.get_allocated_rb()):
            # 计算噪声功率  1个RB, 12个连续的载波, 12 * 15000 = 180000Hz
            white_noise = -174  # -174dBm / Hz
            noise_fig = 1  # dB
            noise_fig = pow(10, noise_fig / 10)  # 线性值
            thermal_noise_pow = pow(10, (white_noise - 30) / 10) * 180000 * noise_fig  # 线性值

            for id in self.__txs_id:
                # 计算接收目标信号功率
                target_tx = dict_id2tx[id]  # 目标发射机
                target_power = target_tx.get_power()  # dBm
                target_power = pow(10, (target_power - 30) / 10)  # W
                target_channel = dict_id2channel[self.get_id()]
                target_link_loss = target_channel.get_link_loss(id)  # dB
                target_gain = pow(10, -target_link_loss / 10)
                receive_target_power = target_power * target_gain

                # 计算接收干扰信号总功率
                receive_inter_power = 0
                for tx_id in dict_id2tx:
                    if tx_id != id:
                        if target_tx.get_allocated_rb()[0] in dict_id2tx[tx_id].get_allocated_rb():
                            inter_tx = dict_id2tx[tx_id]  # 干扰发射机
                            inter_power = inter_tx.get_power()  # dBm
                            inter_power = pow(10, (inter_power - 30) / 10)  # W
                            inter_channel = dict_id2channel[self.get_id()]
                            inter_link_loss = inter_channel.get_link_loss(tx_id)  # dB
                            inter_gain = pow(10, -inter_link_loss / 10)
                            receive_inter_power += inter_power * inter_gain

                sinr = 10 * math.log10(receive_target_power / (receive_inter_power + thermal_noise_pow))
                self.__tx_id2sinr[id] = sinr

            return None

    def comp_q_learning_state(self, target_tx, dict_id2tx, dict_id2channel):  # 计算接收 SINR
        if len(self.get_allocated_rb()):
            # 计算噪声功率  1个RB, 12个连续的载波, 12 * 15000 = 180000Hz
            white_noise = -174  # -174dBm / Hz
            noise_fig = 1  # dB
            noise_fig = pow(10, noise_fig / 10)  # 线性值
            thermal_noise_pow = pow(10, (white_noise - 30) / 10) * 180000 * noise_fig  # 线性值

            # 计算接收目标信号功率
            target_power = target_tx.get_power()  # dBm
            target_power = pow(10, (target_power - 30) / 10)  # W
            target_channel = dict_id2channel[self.get_id()]
            target_link_loss = target_channel.get_link_loss(target_tx.get_id())  # dB
            target_gain = pow(10, -target_link_loss / 10)
            receive_target_power = target_power * target_gain

            # 计算接收干扰信号总功率
            receive_inter_power = 0
            for tx_id in dict_id2tx:
                if tx_id != id:
                    if target_tx.get_allocated_rb()[0] in dict_id2tx[tx_id].get_allocated_rb():
                        inter_tx = dict_id2tx[tx_id]  # 干扰发射机
                        inter_power = inter_tx.get_power()  # dBm
                        inter_power = pow(10, (inter_power - 30) / 10)  # W
                        inter_channel = dict_id2channel[self.get_id()]
                        inter_link_loss = inter_channel.get_link_loss(tx_id)  # dB
                        inter_gain = pow(10, -inter_link_loss / 10)
                        receive_inter_power += inter_power * inter_gain

            sinr = 10 * math.log10(receive_target_power / (receive_inter_power + thermal_noise_pow))

            return sinr

    def get_sinr(self):
        return self.__tx_id2sinr


# 用户类
class User(Interface):
    def __init__(self, i_id, i_type):
        # 调用父类的构造函数
        Interface.__init__(self, i_id, i_type)

        self.__x_point = -1
        self.__y_point = -1
        self.__allocated_rb = []

    def set_location(self, x_point, y_point):
        self.__x_point = x_point
        self.__y_point = y_point

    def get_x_point(self):
        return self.__x_point

    def get_y_point(self):
        return self.__y_point

    def set_allocated_rb(self, rb_id):
        self.__allocated_rb = []
        self.__allocated_rb.append(rb_id)

    def get_allocated_rb(self):
        return self.__allocated_rb

    def update_location(self):
        self.__x_point += random.normalvariate(0, 1)
        self.__y_point += random.normalvariate(0, 1)


# 蜂窝用户类
class CUE(User):
    def __init__(self, i_id, i_type, power=23):
        User.__init__(self, i_id, i_type)
        self.__power = power

    def get_power(self):
        return self.__power


# D2D发射机类
class D2DTx(User):
    def __init__(self, i_id, i_type, power=5):
        User.__init__(self, i_id, i_type)
        self.__rx_id = -1
        self.__power = power
        '''
        power level:
        -10 —— 17dB
        '''

        self.previous_rb = -1
        self.previous_inter = -1
        self.previous_neighbor_1_rb = -1
        self.previous_neighbor_2_rb = -1
        self.previous_neighbor_3_rb = -1
        self.d2d_csi = -1
        self.tx2bs_csi = -1
        # 7维 先前RB, 先前干扰, Rx附近Tx的RB选择(暂定3个), D2D_CSI, D2DTx2BS_CSI
        self.__observation = []

        self.train = False
        self.reward = 0
        self.__action = -1

        self.q_learning_table = QLearningTable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.observation = 0
        self.power_level = -1

    # 配对
    def make_pair(self, rx_id):
        self.__rx_id = rx_id

    def get_power(self):
        return self.__power

    def get_rx_id(self):
        return self.__rx_id

    def update_observation(self, rb_num):
        # 维度 4 * rb_num + 3
        self.__observation = []

        for i in range(rb_num):
            if i == self.previous_rb:
                self.__observation.append(1)
            else:
                self.__observation.append(0)

        self.__observation.append(10**12*self.previous_inter)

        for i in range(rb_num):
            if i == self.previous_neighbor_1_rb:
                self.__observation.append(1)
            else:
                self.__observation.append(0)

        for i in range(rb_num):
            if i == self.previous_neighbor_2_rb:
                self.__observation.append(1)
            else:
                self.__observation.append(0)

        for i in range(rb_num):
            if i == self.previous_neighbor_3_rb:
                self.__observation.append(1)
            else:
                self.__observation.append(0)

        self.__observation.append(self.d2d_csi)
        self.__observation.append(self.tx2bs_csi)

    def choose_action(self, RL, dict_id2rx, rb_num, power_level_num):
        # 根据状态选择行为
        self.update_observation(rb_num)
        state = self.__observation
        action = RL.act(state)
        self.__action = action
        action = int(action)
        rb_id = int(action / power_level_num)
        power_level = action % power_level_num
        self.set_allocated_rb(rb_id)
        self.__power = -10 + power_level * 3

        print('D2DTx ' + str(self.get_id()) +
              ' choose RB: ' + str(rb_id) + ' power: ' + str(self.__power) + ' dB')

        rx = dict_id2rx[self.__rx_id]
        rx.set_allocated_rb(rb_id)

    def choose_action_test(self, RL, dict_id2rx, rb_num, power_level_num):
        # 根据状态选择行为
        self.update_observation(rb_num)
        state = self.__observation
        action = RL.act_test(state)
        self.__action = action
        action = int(action)
        rb_id = int(action / power_level_num)
        power_level = action % power_level_num
        self.set_allocated_rb(rb_id)
        self.__power = -10 + power_level * 3

        rx = dict_id2rx[self.__rx_id]
        rx.set_allocated_rb(rb_id)

    def sa_choose_action(self, RL, dict_id2rx, rb_num, power_level_num):
        # 根据状态选择行为
        self.update_observation(rb_num)
        state = self.__observation
        action = RL.act(state)
        self.__action = action
        rb_id = int(action)
        self.set_allocated_rb(rb_id)

        print('D2DTx ' + str(self.get_id()) +
              ' choose RB: ' + str(rb_id))

        rx = dict_id2rx[self.__rx_id]
        rx.set_allocated_rb(rb_id)

    def sa_choose_action_test(self, RL, dict_id2rx, rb_num, power_level_num):
        # 根据状态选择行为
        self.update_observation(rb_num)
        state = self.__observation
        action = RL.act_test(state)
        self.__action = action
        rb_id = int(action)
        self.set_allocated_rb(rb_id)

        rx = dict_id2rx[self.__rx_id]
        rx.set_allocated_rb(rb_id)

    def learn(self, slot, RL, rb_num):
        state = self.__observation
        self.update_observation(rb_num)
        next_state = self.__observation
        # 存储记忆
        RL.remember(state, self.__action, self.reward, next_state)

        # 当回合数大于32后，每5回合学习1次（先积累一些记忆再开始学习）

        if (slot > 32) and (slot % 5 == 0):
            RL.replay()

    def set_power(self, power_level):
        self.__power = -10 + power_level * 3


# D2D接收机类
class D2DRx(User):
    def __init__(self, i_id, i_type):
        User.__init__(self, i_id, i_type)
        self.__tx_id = -1
        self.__sinr = 0

    # 配对
    def make_pair(self, tx_id):
        self.__tx_id = tx_id

    def get_tx_id(self):
        return self.__tx_id

    def comp_sinr(self, dict_id2tx, dict_id2channel):  # 计算接收 SINR
        if len(self.get_allocated_rb()):
            # 计算噪声功率  1个RB, 12个连续的载波, 12 * 15000 = 180000Hz
            white_noise = -174  # -174dBm / Hz
            noise_fig = 1  # dB
            noise_fig = pow(10, noise_fig / 10)  # 线性值
            thermal_noise_pow = pow(10, (white_noise - 30) / 10) * 180000 * noise_fig  # 线性值

            # 计算接收目标信号功率
            target_tx = dict_id2tx[self.__tx_id]  # 目标发射机
            target_power = target_tx.get_power()  # dBm
            target_power = pow(10, (target_power - 30) / 10)  # W
            target_channel = dict_id2channel[self.get_id()]
            target_link_loss = target_channel.get_link_loss(self.__tx_id)  # dB
            target_gain = pow(10, -target_link_loss / 10)
            receive_target_power = target_power * target_gain

            # 计算接收干扰信号总功率
            receive_inter_power = 0
            for tx_id in dict_id2tx:
                if tx_id != self.__tx_id:
                    if self.get_allocated_rb()[0] in dict_id2tx[tx_id].get_allocated_rb():
                        inter_tx = dict_id2tx[tx_id]  # 干扰发射机
                        inter_power = inter_tx.get_power()  # dBm
                        inter_power = pow(10, (inter_power - 30) / 10)  # W
                        inter_channel = dict_id2channel[self.get_id()]
                        inter_link_loss = inter_channel.get_link_loss(tx_id)  # dB
                        inter_gain = pow(10, -inter_link_loss / 10)
                        receive_inter_power += inter_power * inter_gain

            self.__sinr = 10 * math.log10(receive_target_power / (receive_inter_power + thermal_noise_pow))
            return receive_inter_power

    def get_sinr(self):
        return self.__sinr
