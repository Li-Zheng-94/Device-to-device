from spectrum_allocation_power_control.topology import SingleCell
from spectrum_allocation_power_control.ddqn_keras import DDQNAgent

if __name__ == '__main__':
    slot_num = 2000  # 循环次数
    radius = 500  # m
    cue_num = 20
    d2d_num = 40
    rb_num = 20
    up_or_down_link = 'up'
    d_tx2rx = 10  # m
    power_level_num = 10

    '''
    int(action / power_level_num) = rb_id
    action % power_level_num = power_level
    action = 0 : 选择 0 RB 0 power level
    action = 1 : 选择 0 RB 1 power level
    '''

    single_cell = SingleCell(radius, cue_num, d2d_num, rb_num, up_or_down_link, d_tx2rx, power_level_num)
    single_cell.initial()

    sa_RL = DDQNAgent(4 * rb_num + 3, rb_num)

    for slot in range(slot_num):
        print("********************循环次数: ", slot, " ********************")
        single_cell.sa_train_work(slot, sa_RL)
        single_cell.update()
        if slot != 0 and slot % 1000 == 0:
            sa_RL.save('./save/sa_ddqn', slot)

    sa_RL.save('./save/sa_ddqn', slot_num)
