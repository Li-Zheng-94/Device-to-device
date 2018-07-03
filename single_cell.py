from topology import SingleCell
from DQN import DeepQNetwork

if __name__ == '__main__':
    radius = 500  # m
    cue_num = 10
    d2d_num = 20
    rb_num = 10
    up_or_down_link = 'up'
    d_tx2rx = 10  # m
    single_cell = SingleCell(radius, cue_num, d2d_num, rb_num, up_or_down_link, d_tx2rx)
    single_cell.initial()
    single_cell.work()

    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.01,
                      # output_graph=True
                      )

    # 根据状态选择行为
    action = RL.choose_action(observation)

    # 执行行为，获取下一个状态和奖励
    observation_, reward, done = env.step(action)

    # 存储记忆
    RL.store_transition(observation, action, reward, observation_)
