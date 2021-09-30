'''
Created on Jun 18, 2016

An DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>
- Keep a copy DQN

Command: python .\run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path .\deep_dialog\data\movie_kb.1k.json --dqn_hidden_size 80 --experience_replay_pool_size 1000 --replacement_steps 50 --per_train_epochs 100 --episodes 200 --err_method 2


@author: xiul
'''
import random, copy, json
import pickle
import numpy as np

from collections import deque
from typing import Deque, Dict, List, Tuple

from deep_dialog import dialog_config
from .agent import Agent
from deep_dialog.qlearning import DuellingDQN
from deep_dialog.qlearning import nStepDQN


class Agent_nStep(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())  # 动作集合的大小
        self.slot_cardinality = len(slot_set.keys())  # 槽集合的大小

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)
        self.available_actions = range(self.num_actions)  # [0,1,2,...,num_actions]
        self.new_actions = range(self.num_actions)  # [0,1,2,...,num_actions]

        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']

        # n-step init
        self.n_step = 3
        self.n_step_pool = deque(maxlen=self.n_step)
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1, t ,u_t>
        self.experience_replay_pool_from_model = []  # 存放"世界模型生成的经验"的回放缓存池，B^s

        self.epsilon = params['epsilon']
        self.gamma = params.get('gamma', 0.9)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.warm_start = params.get('warm_start', 0)
        self.max_turn = params['max_turn'] + 5

        self.refine_state = True
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        if self.refine_state:
            self.state_dimension = 213

        self.dqn = nStepDQN(self.state_dimension, self.hidden_size, self.num_actions, self.n_step)
        self.clone_dqn = copy.deepcopy(self.dqn)

        self.predict_mode = params.get('predict_mode', False)

        self.small_buffer = False
        self.cur_bellman_err = 0

        # replay buffer settings
        self.model_type = params['model_type']  # DQN DDQ D3Q
        self.size_unit = params['buffer_size_unit']
        self.planning_steps = params['planning_steps']
        # the replay buffer size also follow the planning step concept?
        if params['planning_step_to_buffer']:
            if self.model_type == "DQN":
                self.max_user_buffer_size = self.size_unit * (self.planning_steps + 1)
                self.max_world_model_buffer_size = 0
            else:
                # DDQ, D3Q
                self.max_user_buffer_size = self.size_unit
                self.max_world_model_buffer_size = self.size_unit * self.planning_steps
        else:
            if self.model_type == "DQN":
                self.max_user_buffer_size = self.size_unit
                self.max_world_model_buffer_size = 0
            else:
                # DDQ, D3Q
                self.max_user_buffer_size = self.size_unit
                self.max_world_model_buffer_size = self.size_unit
        '''
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            # self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            # self.clone_dqn = copy.deepcopy(self.dqn)
            self.dqn.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2
        '''

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):
        """ DQN: Input state, output action """
        self.representation = self.prepare_state_representation(state)  # 将状态转换成可输入的状态表示
        self.action = self.run_policy(self.representation)  # 基于状态表示，根据epsilon-greedy策略得到动作
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])  # 根据动作填槽
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        agent_last = state['agent_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #  Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # turn_rep = np.zeros((1,1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        # ########################################################################
        # #   Representation of KB results (scaled counts)
        # ########################################################################
        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #    if slot in self.slot_set:
        #        kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        #
        # ########################################################################
        # #   Representation of KB results (binary)
        # ########################################################################
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #    if slot in self.slot_set:
        #        kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))
        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

        self.final_representation = np.hstack([
            user_act_rep,
            user_inform_slots_rep,
            user_request_slots_rep,
            agent_act_rep,
            agent_inform_slots_rep,
            agent_request_slots_rep,
            current_slots_rep,
            turn_rep,
            turn_onehot_rep
        ])

        return self.final_representation

    # NoisyNet version
    def run_policy(self, representation):
        """ no epsilon greedy action selection """
        if self.warm_start == 1:
            if len(self.experience_replay_pool) > self.experience_replay_pool_size:  # 若回放缓存池溢出
                self.warm_start = 2
            return self.rule_policy()  # 返回的是索引
        else:  # 若不是热启动阶段，则基于DQN选择动作
            return self.available_actions[
                np.argmax(self.dqn.predict(representation)[self.available_actions])
            ]

    # 规则策略
    def rule_policy(self):
        """ Rule Policy """

        # 若当前槽位未填满
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1
            # 继续请求信息
            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}

        # 告知信息
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1

        # 表示感谢
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)  # 返回动作所对应的索引

    # 返回给定动作所对应的索引（被函数rule_policy调用）
    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):  # 在self.feasible_actions中查找动作
            if act_slot_response == action:  # 若动作吻合则返回对应的索引
                return i
        print(act_slot_response)  # 打印该动作
        raise Exception("action index not found")  # 查找不到则抛出异常，返回None
        return None

    def _n_step_info(self, n_step_buffer, gamma):
        reward, s_prime, done = n_step_buffer[-1][2:-1]  # get last example info
        for example in reversed(list(n_step_buffer)[:-1]):  # iterate over first two example
            r, s_p, t = example[2:-1]
            reward = r + gamma * reward * (1 - t)
            if t:
                s_prime, done = (s_p, t)
            else:
                s_prime, done = (s_prime, done)
        return reward, s_prime, done

    # 将经验放进回放缓存池（该函数在DM模块中被调用）
    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
        """ Register feedback from the environment, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)  # 对当前状态进行表示
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)  # 对下一时间步的状态进行表示
        st_user = self.prepare_state_representation(s_tplus1)  # 对用户状态进行表示
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)

        # 根据训练/预测模式、来自世界模型与否，相应的存放经验
        if self.predict_mode == False:  # 训练模式
            if self.warm_start == 1:  # 热启动阶段
                self.experience_replay_pool.append(training_example)

        else:  # 预测模式
            if not from_model:  # 真实经验
                # n-step
                if self.n_step > 1:
                    self.n_step_pool.append(training_example)
                    if len(self.n_step_pool) < self.n_step:
                        return ()
                    print("n_step_pool length: {}".format(len(self.n_step_pool)))
                    rew, s_prime, done = self._n_step_info(self.n_step_pool, self.gamma)
                    training_example = (state_t_rep, action_t, rew, s_prime, done, st_user)
                    self.experience_replay_pool.append(training_example)
                # 1-step
                else:
                    self.experience_replay_pool.append(training_example)

            else:  # 模拟经验（来自于世界模型）
                # n-step
                if self.n_step > 1:
                    self.n_step_pool.append(training_example)
                    if len(self.n_step_pool) < self.n_step:
                        return ()
                    print("n_step_pool length: {}".format(len(self.n_step_pool)))
                    rew, s_prime, done = self._n_step_info(self.n_step_pool, self.gamma)
                    training_example = (state_t_rep, action_t, rew, s_prime, done, st_user)
                    self.experience_replay_pool_from_model.append(training_example)
                # 1-step
                else:
                    self.experience_replay_pool_from_model.append(training_example) # 放入世界模型池中

        # 若溢出，则保留最新经验
        if len(self.experience_replay_pool) > self.max_user_buffer_size:
            self.experience_replay_pool = self.experience_replay_pool[-self.max_user_buffer_size:]

        if len(self.experience_replay_pool_from_model) > self.max_world_model_buffer_size:
            self.experience_replay_pool_from_model = self.experience_replay_pool_from_model[
                                                     -self.max_world_model_buffer_size:]

    # run over the whole replay buffer 汇总经验，根据batch_size打包预处理后进行训练
    def train(self, batch_size=16, num_iter=1, controller=0, use_real_example=True):
        """ Train DQN with experience replay """
        self.cur_bellman_err = 0
        self.cur_bellman_err_planning = 0
        running_expereince_pool = self.experience_replay_pool + self.experience_replay_pool_from_model  # 总经验池

        for iter in range(num_iter):
            for _ in range(len(running_expereince_pool) // (batch_size)):  # 迭代batch数量次

                batch = [random.choice(running_expereince_pool) for _ in range(batch_size)]  # 从总经验池中随机挑出batch_size个经验
                np_batch = []  # each example in a batch: [s, a, r, s_prime, term]，大小为(5,16)
                for x in range(5):
                    v = []
                    for i in range(len(batch)):  # 内循环16次，外循环5次
                        v.append(batch[i][x])  # 纵向抽取16条经验中的每一个属性放入到v[]中
                    np_batch.append(np.vstack(
                        v))  # np_batch列表中的每一个元素为经验五元组中的一个项，如s、a、r、s_prime、term；其中列表元素（如np_batch[0]为状态s）的大小为(16,1)，
                self.dqn.singleBatch(np_batch)  # 调用DQN进行训练

            if len(self.experience_replay_pool) != 0:
                print(
                    "cur bellman err %.4f, experience replay pool %s, model replay pool %s, cur bellman err for planning %.4f" % (
                        float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
                        len(self.experience_replay_pool), len(self.experience_replay_pool_from_model),
                        self.cur_bellman_err_planning))

    # train specific number of batches
    # def train(self, batch_size=16, num_iter=1, planning=False, controller=0, use_real_example=True):
    def train_one_iter(self, batch_size=16, num_batches=1, planning=False, controller=0, use_real_example=True):
        """ Train DQN with experience replay """
        self.cur_bellman_err = 0
        self.cur_bellman_err_planning = 0
        running_expereince_pool = self.experience_replay_pool + self.experience_replay_pool_from_model
        for _ in range(num_batches):
            batch = [random.choice(self.experience_replay_pool) for i in range(batch_size)]
            np_batch = []
            for x in range(5):
                v = []
                for i in range(len(batch)):
                    v.append(batch[i][x])
                np_batch.append(np.vstack(v))

            batch_struct = self.dqn.singleBatch(np_batch)
        if len(self.experience_replay_pool) != 0:
            print("cur bellman err %.4f, experience replay pool %s, cur bellman err for planning %.4f" % (
                float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
                len(self.experience_replay_pool), self.cur_bellman_err_planning))

    def set_user_planning(self, user_planning):
        self.user_planning = user_planning

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path,))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path,))
            print(e)

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']

        print("trained DQN Parameters:", json.dumps(trained_file['params'], indent=2))
        return model

    def save_dqn(self, path):
        # return self.dqn.unzip()
        self.dqn.save_model(path)

    def load_dqn(self, params):
        self.dqn.load(params)

    ################################################################################
    #    not-be-used functions
    ################################################################################
    # copy actions
    # def set_actions(self, the_actions):
    #     self.available_actions = copy.deepcopy(the_actions)
    #     self.new_actions = copy.deepcopy(the_actions)
    #
    # def add_actions(self, new_actions):
    #     self.new_actions = copy.deepcopy(new_actions)
    #     self.available_actions += new_actions
    #     # self.q_network.add_actions(new_actions)
