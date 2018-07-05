from topology import SingleCell
from DQN import DeepQNetwork

if __name__ == '__main__':
    slot_num = 1000  # 循环次数
    radius = 500  # m
    cue_num = 10
    d2d_num = 20
    rb_num = 10
    up_or_down_link = 'up'
    d_tx2rx = 10  # m

    single_cell = SingleCell(radius, cue_num, d2d_num, rb_num, up_or_down_link, d_tx2rx)
    single_cell.initial()

    RL = DeepQNetwork(rb_num, 7,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.01,
                      # output_graph=True
                      )

    for slot in range(slot_num):
        print("循环次数：", slot)
        single_cell.work(slot, RL)
        single_cell.update()

