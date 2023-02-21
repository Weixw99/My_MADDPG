import numpy as np
import tensorflow as tf
import time
import pickle
import os
import common.utils as util
from agent import make_env, get_trainers

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
model_path = curr_path + '/models/'


class Parameters:
    def __init__(self):
        # Environment
        self.scenario = 'simple_v0'  # 定义要使用 MPE 中的哪个环境
        self.algo_name = 'ma-ddpg'  # 算法名称
        self.device = 'cuda' if tf.test.is_gpu_available() else 'cpu'  # 检测GPU
        self.episodes_num = 6000  # 训练的回合数
        self.episodes_len = 300  # 每回合步数
        self.adversaries_num = 0  # 环境中的对手数量
        self.good_policy = 'ma-ddpg'  # 用于环境中“良好”（非对手）策略的算法
        self.adv_policy = 'ma-ddpg'  # 用于环境中对手策略的算法

        # Core training parameters
        self.lr = 1e-2  # 学习率
        self.gamma = 0.95  # 折扣因子
        self.batch_size = 1024  # 批量大小
        self.units_num = 64  # MLP 中的单元数

        # Checkpointing
        self.exp_name = ''  # 实验名称，用作保存所有结果的文件名
        self.save_dir = model_path  # 保存中间训练结果和模型的目录
        self.save_rate = 100  # 每次完成此数量的训练时都会保存模型
        self.load_dir = ''  # 从中加载训练状态和模型的目录

        # Evaluation
        self.restore = False  # 恢复存储在load-dir（或save-dir如果未load-dir 提供）中的先前训练状态，并继续训练
        self.display = False  # 在屏幕上显示存储在load-dir（或save-dir如果没有load-dir 提供）中的训练策略，但不继续训练
        self.benchmark = False  # 对保存的策略运行基准评估，将结果保存到benchmark-dir文件夹
        self.benchmark_iter = 100000  # 运行基准测试的迭代次数
        self.benchmark_dir = './benchmark_files/'  # 保存基准数据的目录
        self.plots_dir = './learning_curves/'  # 保存训练曲线的目录


def train(parameters):
    with util.single_threaded_session():
        # 创建环境
        env = make_env(parameters.scenario, parameters.benchmark)
        # 创建agent
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        adversaries_num = min(env.n, parameters.adversaries_num)
        trainers = get_trainers(env, adversaries_num, obs_shape_n, parameters)
        print('Use scenario:{},algo:{},device:{}'.format(parameters.scenario, parameters.algo_name, parameters.device))

        # 初始化
        util.initialize()

        parameters.save_dir = os.path.join(parameters.save_dir, parameters.scenario)
        if not os.path.exists(parameters.save_dir):
            os.makedirs(parameters.save_dir)
        total_files = len([file for file in os.listdir(parameters.save_dir)])
        # 判断是否加载上次训练好的模型
        if parameters.display or parameters.restore or parameters.benchmark:
            if parameters.load_dir == "":
                parameters.save_dir = os.path.join(parameters.save_dir, f'{total_files}' + '/')
                parameters.load_dir = parameters.save_dir
            print('Loading previous state...')
            util.load_state(parameters.load_dir)  # 加载模型
        else:
            parameters.save_dir = os.path.join(parameters.save_dir, f'{total_files + 1}' + '/')
            os.makedirs(parameters.save_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.compat.v1.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        print('Starting iterations...')
        while True:
            # 获取action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= parameters.episodes_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):  # 奖励累计
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
            # increment global step counter
            train_step += 1
            # for benchmarking learned policies
            if parameters.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > parameters.benchmark_iters and (done or terminal):
                    file_name = parameters.benchmark_dir + parameters.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue
            # 显示
            if parameters.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

                # print(loss)
            # save model, display training output
            if terminal and (len(episode_rewards) % parameters.save_rate == 0):
                util.save_state(parameters.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if adversaries_num == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-parameters.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-parameters.save_rate:]),
                        [np.mean(rew[-parameters.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-parameters.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-parameters.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > parameters.episodes_num:
                rew_file_name = parameters.plots_dir + parameters.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = parameters.plots_dir + parameters.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    parameter = Parameters()
    train(parameter)
