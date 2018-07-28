from spectrum_allocation.topology import SingleCell
# from DQN import DeepQNetwork
# from dqn_keras import DQNAgent
from spectrum_allocation.ddqn_keras import DDQNAgent

if __name__ == '__main__':
    slot_num = 50000  # 循环次数
    radius = 500  # m
    cue_num = 20
    d2d_num = 40
    rb_num = 20
    up_or_down_link = 'up'
    d_tx2rx = 30  # m

    single_cell = SingleCell(radius, cue_num, d2d_num, rb_num, up_or_down_link, d_tx2rx)
    single_cell.initial()

    # RL = DQNAgent(4 * rb_num + 3, rb_num)
    RL = DDQNAgent(4 * rb_num + 3, rb_num)

    for slot in range(slot_num):
        print("********************循环次数: ", slot, " ********************")
        # single_cell.random_allocation_work(slot)
        single_cell.work(slot, RL)
        single_cell.update()
        if slot != 0 and slot % 2500 == 0:
            RL.save('./spectrum_allocation/save/ddqn', slot)

    RL.save('./save/ddqn', slot_num)
    # single_cell.plot()
    # single_cell.save_data()


