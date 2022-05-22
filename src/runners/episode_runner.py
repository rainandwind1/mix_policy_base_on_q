import enum
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import random
import pandas as pd
import collections
import os
import datetime
from collections import deque
from sklearn.cluster import SpectralClustering, KMeans

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        # total
        self.t_env = 0
        self.t_macro = 0
        self.t_move = 0
        self.t_action = 0

        # epi count
        self.macro_t = 0
        self.move_t = 0
        self.action_t = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        # for test goal random goal candidate
        self.goal_box = [[np.ones(self.env.get_goal_size()) for _ in range(self.args.goal_num)] for _ in range(self.env.n_agents)]

        # reward assistant
        self.reward_assistant = Agent_assistant(args = args)

    #(macro_scheme=macro_scheme, move_scheme=move_scheme, action_scheme=action_scheme, groups=groups, move_preprocess=move_preprocess, action_preprocess=action_preprocess, macro_preprocess=macro_preprocess, move_mac=move_mac, action_mac=action_mac, macro_mac=macro_mac)
    def setup(self, scheme, groups, preprocess, mac):
        
        # 2021/07/26
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        
        # controller
        self.map_data = [[] for _ in range(self.env.n_agents)]
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()

        # episode step counter
        self.t = 0

    def goal_id_to_goal(self, goal_ids):
        goals = []
        for idx, goal_id in enumerate(goal_ids):
            goals.append(self.goal_box[idx][int(goal_id)])
        return goals

    def get_avail_macro_ids(self, all_actions):
        avail_macro_ids_ls = []
        for i in range(self.env.n_agents):
            agent_avail_action_ls = all_actions[i]
            res = [0, 0]
            if 1 in agent_avail_action_ls[:self.args.a_move_size]:
                res[0] = 1
            if 1 in agent_avail_action_ls[self.args.a_move_size:]:
                res[1] = 1
            avail_macro_ids_ls.append(res)
            # print(avail_macro_ids_ls)
        return avail_macro_ids_ls

    def cal_intrinsic_reward(self, goal_feats, goals):
        intrinsic_reward_ls = [[]]
        for goal_feat, goal in zip(goal_feats, goals):
            intrinsic_reward_ls[0].append(-np.linalg.norm(goal_feat - goal))
        return intrinsic_reward_ls

    def goal_exploration(self, logger, test_mode=False):
        logger.console_logger.info("Begin exploration goals: Random walk!")
        if test_mode:
            pass
        
        t_random = 0
        epi_random = 0.
        end = False
        goal_obs_feats_buffer = [[] for _ in range(self.env.n_agents)]
        good_goal_obs_feats_buffer = [[] for _ in range(self.env.n_agents)]
        env_info = self.get_env_info()
        while t_random < self.args.random_walk:
            epi_random += 1
            self.env.reset()
            end = False
            while not end:  
                # random action choose and execute
                actions = []
                for idx in range(self.env.n_agents):
                    avail_mask = self.env.get_avail_agent_actions(idx)
                    candidate_action = range(env_info["n_actions"])
                    candidate_action = [i for i in candidate_action if avail_mask[i] > 0]
                    action = random.sample(candidate_action, 1)[0]
                    actions.append(action)
                
                reward, terminated, info = self.env.step(actions)

                # s_next
                goal_obs_feats_ls = self.env.get_goal_feats()
                for agent_i in range(self.env.n_agents):
                    if reward <= 0:
                        goal_obs_feats_buffer[agent_i].append((goal_obs_feats_ls[agent_i], reward))
                    else:
                        good_goal_obs_feats_buffer[agent_i].append((goal_obs_feats_ls[agent_i], reward))

                # update counter
                t_random += 1
                end = terminated

                if (t_random + 1) % 10000 == 0:
                    logger.console_logger.info("{} Step execute!".format(t_random + 1))
        
        logger.console_logger.info("Update goal box!")
        self.update_goal_box(goal_obs_feats_buffer, good_goal_obs_feats_buffer)
        logger.console_logger.info("End exploration goals: Save goals to excel!")
    

    def update_goal_box(self, goal_obs_feats_buffer, good_goal_obs_feats_buffer):
        # kmeans + anomaly detection

        # anomaly detection
        for idx, goal_obs_trans in enumerate(good_goal_obs_feats_buffer):
            goal_obs_trans = sorted(goal_obs_trans, key = lambda item:item[1], reverse=True)
            goal_obs_data = [trans[0] for trans in goal_obs_trans if trans[1] > 0]
            if len(goal_obs_data) > self.args.goal_num:
                self.goal_box[idx] = []
                count = 0
                # 重复检测
                while len(self.goal_box[idx]) < self.args.goal_num:
                    if list(goal_obs_data[count]) not in self.goal_box[idx]:
                        self.goal_box[idx].append(list(goal_obs_data[count]))
                    count += 1
            else:
                self.goal_box[idx] = []
                count = 0
                # 重复检测
                while count < len(goal_obs_data):
                    if list(goal_obs_data[count]) not in self.goal_box[idx]:
                        self.goal_box[idx].append(list(goal_obs_data[count]))
                    count += 1
        
        if len(self.goal_box[idx]) < self.args.goal_num:
            # kmeans
            for idx, goal_obs_trans in enumerate(goal_obs_feats_buffer):
                goal_obs_data = [trans[0] for trans in goal_obs_trans]
                s = KMeans(n_clusters=self.args.goal_num, random_state=0).fit(goal_obs_data)
                goal_obs_core = s.cluster_centers_
                #goal_obs_core = self.kmeans(np.stack(goal_obs_data), self.args.goal_num)
                for i in range(goal_obs_core.shape[0]):
                    if list(goal_obs_core[i]) not in self.goal_box[idx]:
                        self.goal_box[idx].append(list(goal_obs_core[i]))
                # 限长
                self.goal_box[idx] = self.goal_box[idx][:self.args.goal_num]

        self.save_goals()

    def save_goals(self):
        writer = pd.ExcelWriter('./goal.xlsx')		# 写入Excel文件
        
        for agent_i, agent_goal_ls in enumerate(self.goal_box):
            data = pd.DataFrame(np.array(agent_goal_ls))
            data.to_excel(writer, 'page_' + '{}'.format(agent_i), float_format='%.5f')		# ‘page_1’是写入excel的sheet名
        
        writer.save()
        writer.close()
                        
    
    def kmeans(self, ds, k):
        """ k-means聚类算法
        k       - 指定分簇数量
        ds      - ndarray(m, n)，m个样本的数据集，每个样本n个属性值
        """
        
        m, n = ds.shape # m：样本数量，n：每个样本的属性值个数
        result = np.empty(m, dtype=np.int) # m个样本的聚类结果
        cores = ds[np.random.choice(np.arange(m), k, replace=False)] # 从m个数据样本中不重复地随机选择k个样本作为质心
        count = 0
        while True: # 迭代计算
            d = np.square(np.repeat(ds, k, axis=0).reshape(m, k, n) - cores)
            distance = np.sqrt(np.sum(d, axis=2)) # ndarray(m, k)，每个样本距离k个质心的距离，共有m行
            index_min = np.argmin(distance, axis=1) # 每个样本距离最近的质心索引序号
            
            if (index_min == result).all(): # 如果样本聚类没有改变
                return result, cores # 则返回聚类结果和质心数据
            
            result[:] = index_min # 重新分类
            for i in range(k): # 遍历质心集
                items = ds[result==i] # 找出对应当前质心的子样本集
                cores[i] = np.mean(items, axis=0) # 以子样本集的均值作为当前质心的位置

    def get_avail_micro_actions(self, all_avail_actions, macro_ids):
        avail_actions = []
        for idx, all_avail_action in enumerate(all_avail_actions):
            if int(macro_ids[idx]) == 0:
                avail_actions.append(all_avail_action[:self.args.a_move_size] + [0] * (len(all_avail_action) - self.args.a_move_size))
            elif int(macro_ids[idx]) == 1:
                avail_actions.append([0] * self.args.a_move_size + all_avail_action[self.args.a_move_size:])
        return avail_actions

    def get_avail_move_a_actions(self, all_avail_actions):
        avail_move_ids = []
        avail_a_ids = []
        for idx, all_avail_action in enumerate(all_avail_actions):
            avail_move_ids.append(all_avail_action[:self.args.a_move_size])
            avail_a_ids.append(all_avail_action[self.args.a_move_size:])
            if 1 not in avail_a_ids[-1]:
                avail_a_ids[-1] += [1]     # no op added for action policy action selection
            else:
                avail_a_ids[-1] += [0]
        return avail_move_ids, avail_a_ids

    def get_actions(self, move_actions, a_actions, macro_ids):
        actions = []
        for idx, macro_id in enumerate(macro_ids):
            if int(macro_id) == 0:
                actions.append(move_actions[idx])
            elif int(macro_id) == 1:
                actions.append(a_actions[idx] + self.args.a_move_size)
        return actions

    def decomposing_reward(self, reward, macro_ids):
        count = 0
        for action in macro_ids:
            if int(action) == 0:
                count += 1
        reward_move = (count / len(macro_ids)) * reward
        reward_a = reward - reward_move
    
        return reward_move, reward_a
        



    def run(self, test_mode=False):
        self.reset()

        # for debug
        # state_debug = self.env.get_state()
        # obs_debug = self.env.get_obs()

        terminated = False
        episode_return = 0
        episode_weights = []

        # micro mac initialize hidden weight
        self.mac.init_hidden(batch_size=self.batch_size)
        

        while not terminated:

            # obs divided into 4 items: move feats、enemy feats、ally feats、own feats
            agent_obs_feats = self.env.get_obs_feats()

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "adjacency_matrix":[self.env.get_adjacency_matrix()],
                "move_feats":[agent_obs_feats[0]],
                "enemy_feats":[agent_obs_feats[1]],
                "ally_feats":[agent_obs_feats[2]],
                "own_feats":[agent_obs_feats[3]],
                "macro_inputs":[self.env.get_obs_feats_normal()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, embedding_weights = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # print(actions.shape)
            # save weights
            episode_weights.append(embedding_weights)
            
            reward, terminated, env_info = self.env.step(actions[0])
        
            episode_return += reward
            assistant_reward = self.reward_assistant.get_assistant_reward(actions[0].cpu().numpy())
            reward_ = max(reward, assistant_reward)

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)]
            }

            self.reward_assistant.save_data((actions[0].cpu().numpy(), reward))

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        # save 
        if test_mode:
            np.save(os.path.join(self.args.weight_save_path, '{}_weights_{}.npy'.format(self.args.env, datetime.datetime.now().strftime("%H_%M_%Y"))), np.stack(episode_weights))

        agent_obs_feats = self.env.get_obs_feats()
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "adjacency_matrix":[self.env.get_adjacency_matrix()],
            "move_feats":[agent_obs_feats[0]],
            "enemy_feats":[agent_obs_feats[1]],
            "ally_feats":[agent_obs_feats[2]],
            "own_feats":[agent_obs_feats[3]],
            "macro_inputs":[self.env.get_obs_feats_normal()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, embedding_weights = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

class Agent_assistant():
    def __init__(self, args):
        self.args = args
        self.assistant_mem_size = 10000
        self.mem_buffer = deque(maxlen=self.assistant_mem_size)
        self.cnt = 0
        self.class_num = 4
        self.update_interval = 1000
        self.cluster_reward_ls = [[] for _ in range(self.class_num)]
        self.cluster_avg_reward = [0. for _ in range(self.class_num)]
    
    def save_data(self, data):
        self.mem_buffer.append(data)
    
    # kmeans
    def update_center(self, data):
        action_cluster = KMeans(n_clusters=self.class_num, random_state=0).fit(data)
        self.action_cluster_center = action_cluster.cluster_centers_
        self.labels = action_cluster.labels_
        self.update_cluster_avg_reward()

    def package_data(self):
        action_ls = []
        reward_ls = []
        for actions, reward in self.mem_buffer:
            action_ls.append(actions)
            reward_ls.append(reward)
        return np.stack(action_ls), reward_ls

        # linear distance
        # np.linalg.norm(self.action_data[idx] - self.action_cluster_center[self.labels[idx]])

    def update_cluster_avg_reward(self):
        for idx, reward in enumerate(self.reward_data):
            self.cluster_reward_ls[self.labels[idx]].append(reward / np.exp(np.count_nonzero(self.action_data[idx] != self.action_cluster_center[self.labels[idx]])))
        for i in range(self.class_num):
            self.cluster_avg_reward[i] = sum(self.cluster_reward_ls[i]) / len(self.cluster_reward_ls[i])

    def get_closest_cluster(self, actions):
        distance_to_cluster = np.zeros(self.class_num)
        for i in range(self.class_num):
            distance_to_cluster[i] = np.linalg.norm(actions - self.action_cluster_center[i])
        return np.argmin(distance_to_cluster)


    def get_assistant_reward(self, action_list):
        if len(self.mem_buffer) <= (self.assistant_mem_size - self.update_interval):
            return 0.
        assistant_reward = 0.
        if self.cnt % self.update_interval == 0:
            self.action_data, self.reward_data = self.package_data() 
            self.update_center(self.action_data)
        self.cnt += 1
        return self.cluster_avg_reward[self.get_closest_cluster(action_list)]

