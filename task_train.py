import numpy as np
import tensorflow as tf
import time
import pickle
import os
import common.utils as util
from agent import make_env, get_trainers

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
parent_path = os.path.dirname(parent_path) + '\\models\\'
print('Target Surrounded')


class Parameters:
    def __init__(self):
        # Environment
        self.scenario = 'simple_v0'
        self.algo_name = 'maddpg'  # 算法名称
        self.device = 'cuda' if tf.test.is_gpu_available() else 'cpu'  # 检测GPU
        self.episodes_num = 6000  # 训练的回合数
        self.episodes_len = 300  # 每回合步数
        self.adversaries_num = 50  # 对手的数量
        self.good_policy = 'maddpg'
        self.adv_policy = 'maddpg'

        # Core training parameters
        self.lr = 1e-2  # learning rate for Adam optimizer
        self.gamma = 0.95  # 折扣因子
        self.batch_size = 1024  # number of episodes to optimize at the same time
        self.units_num = 64  # number of units in the mlp

        # Checkpointing
        self.exp_name = ''  # name of the experiment
        self.save_dir = parent_path  # directory in which training state and model should be saved
        self.save_rate = 1  # save model once every time this many episodes are completed
        self.load_dir = ''  # directory in which training state and model are loaded

        # Evaluation
        self.restore = False
        self.display = False
        self.benchmark = False
        self.benchmark_iter = 100000  # number of iterations run for benchmarking
        self.benchmark_dir = './benchmark_files/'  # directory where benchmark data is saved
        self.plots_dir = './learning_curves/'  # directory where plot data is saved


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

        # 判断是否加载上次训练好的模型
        if parameters.load_dir == "":
            parameters.load_dir = parameters.save_dir
        if parameters.display or parameters.restore or parameters.benchmark:
            print('Loading previous state...')
            util.load_state(parameters.load_dir)  # 加载模型

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
