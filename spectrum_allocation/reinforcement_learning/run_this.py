from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # 初始化状态
        observation = env.reset()

        while True:
            # 更新环境
            env.render()

            # 根据状态选择行为
            action = RL.choose_action(observation)

            # 执行行为，获取下一个状态和奖励
            observation_, reward, done = env.step(action)

            # 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 当回合数大于200后，每5回合学习1次（先积累一些记忆再开始学习）
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 更新状态
            observation = observation_

            if done:
                break
            step += 1

    print("game over")
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.01,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()

