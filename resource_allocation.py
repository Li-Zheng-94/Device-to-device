from device import *


def random_allocation(dict_id2tx, dict_id2rx, rb_num):
    for rx_id in dict_id2rx:
        tx_id = dict_id2rx[rx_id].get_tx_id()
        if type(tx_id) == int:  # D2D
            rb_id = int(rb_num * random.random())
            dict_id2rx[rx_id].set_allocated_rb(rb_id)
            dict_id2tx[tx_id].set_allocated_rb(rb_id)
            print("接收机ID: ", rx_id, " RB ID: ", rb_id)
        else:
            rb_id = 0
            for id in tx_id:
                dict_id2rx[rx_id].set_allocated_rb(rb_id)
                dict_id2tx[id].set_allocated_rb(rb_id)
                print("接收机ID: ", rx_id, " RB ID: ", rb_id)
                rb_id += 1
