from spectrum_allocation.topology import SingleCell
from spectrum_allocation.ddqn_keras import DDQNAgent

if __name__ == '__main__':
    slot_num = 1000  # 循环次数
    radius = 500  # m
    cue_num = 20
    d2d_num = 40
    rb_num = 20
    up_or_down_link = 'up'
    d_tx2rx = 30  # m

    single_cell = SingleCell(radius, cue_num, d2d_num, rb_num, up_or_down_link, d_tx2rx)
    single_cell.initial()

    RL = DDQNAgent(4 * rb_num + 3, rb_num)
    # load weights
    RL.load('./save/ddqn40000.h5')

    for slot in range(slot_num):
        print("********************循环次数: ", slot, " ********************")
        single_cell.random_allocation_work(slot)
        single_cell.rl_test_work(slot, RL)
        single_cell.update()

    single_cell.save_data()

