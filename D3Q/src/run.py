import argparse, json, copy, os
import pickle
import numpy
import random

from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, RequestBasicsAgent, \
    AgentDQN, Agent_nStep, Agent_NoNStep, Agent_NoNoisyNet, Agent_NoDueling, Agent_NoDDQN
from deep_dialog.usersims import RuleSimulator, ModelBasedSimulator
from deep_dialog.controller import Discriminator

from deep_dialog import dialog_config

from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg


# <editor-fold desc="用于读取文件的辅助函数">
def convertFile(originPath):
    origin_str = os.path.splitext(originPath)
    destiPath = origin_str[0] + "_new_" + origin_str[1]
    outsize = 0
    with open(originPath, 'rb') as f:
        content = f.read()
        with open(destiPath, 'wb') as output:
            print("writing to: ", destiPath)
            for line in content.splitlines():
                outsize += len(line) + 1
                output.write(line + str.encode('\n'))
    print("successfully! ")
    return destiPath


# </editor-fold>

# <editor-fold desc="总参数设置">
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dict_path', dest='dict_path', type=str, default='./deep_dialog/data/dicts.v3.p',
                        help='path to the .json dictionary file')
    parser.add_argument('--movie_kb_path', dest='movie_kb_path', type=str, default='./deep_dialog/data/movie_kb.1k.p',
                        help='path to the movie kb .json file')
    parser.add_argument('--act_set', dest='act_set', type=str, default='./deep_dialog/data/dia_acts.txt',
                        help='path to dia act set; none for loading from labeled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str, default='./deep_dialog/data/slot_set.txt',
                        help='path to slot set; none for loading from labeled file')
    parser.add_argument('--goal_file_path', dest='goal_file_path', type=str,
                        default='./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p',
                        help='a list of user goals')
    parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str,
                        default='./deep_dialog/data/dia_act_nl_pairs.v6.json',
                        help='path to the pre-defined dia_act&NL pairs')

    parser.add_argument('--max_turn', dest='max_turn', default=20, type=int,
                        help='maximum length of each dialog (default=20, 0=no maximum length)')
    parser.add_argument('--episodes', dest='episodes', default=1, type=int,
                        help='Total number of episodes to run (default=1)')
    parser.add_argument('--slot_err_prob', dest='slot_err_prob', default=0.05, type=float,
                        help='the slot err probability')
    parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0, type=int,
                        help='slot_err_mode: 0 for slot_val only; 1 for three errs')
    parser.add_argument('--intent_err_prob', dest='intent_err_prob', default=0.05, type=float,
                        help='the intent err probability')

    parser.add_argument('--agt', dest='agt', default=10, type=int,
                        help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
    parser.add_argument('--usr', dest='usr', default=0, type=int,
                        help='Select a user simulator. 0 is a Frozen user simulator.')

    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.00,
                        help='Epsilon to determine stochasticity of epsilon-greedy agent policies')

    # load NLG & NLU model
    parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str,
                        default='./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p',
                        help='path to model file')
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str,
                        default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p',
                        help='path to the NLU model file')

    parser.add_argument('--act_level', dest='act_level', type=int, default=0,
                        help='0 for dia_act level; 1 for NL level')
    parser.add_argument('--run_mode', dest='run_mode', type=int, default=0,
                        help='run_mode: 0 for default NL; 1 for dia_act; 2 for both; 3 for no display (i.e. training)')
    parser.add_argument('--auto_suggest', dest='auto_suggest', type=int, default=0,
                        help='0 for no auto_suggest; 1 for auto_suggest')
    parser.add_argument('--cmd_input_mode', dest='cmd_input_mode', type=int, default=0,
                        help='run_mode: 0 for NL; 1 for dia_act')

    # RL agent parameters
    parser.add_argument('--experience_replay_pool_size', dest='experience_replay_pool_size', type=int, default=1000,
                        help='the size for experience replay')
    parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=60,
                        help='the hidden size for DQN')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
    parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False, help='predict model for DQN')
    parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=50,
                        help='the size of validation set')
    parser.add_argument('--warm_start', dest='warm_start', type=int, default=1,
                        help='0: no warm start; 1: warm start for training')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=100,
                        help='the number of epochs for warm start')
    parser.add_argument('--planning_steps', dest='planning_steps', type=int, default=4,
                        help='the number of planning steps')

    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None,
                        help='the path for trained model')
    parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str,
                        default='./deep_dialog/checkpoints/', help='write model to disk')
    parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10,
                        help='number of epochs for saving model')

    parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.6,
                        help='the threshold for success rate')

    parser.add_argument('--split_fold', dest='split_fold', default=5, type=int,
                        help='the number of folders to split the user goal')
    parser.add_argument('--learning_phase', dest='learning_phase', default='all', type=str,
                        help='train/test/all; default is all')
    parser.add_argument('--grounded', dest='grounded', type=int, default=False, help='planning with ground truth')
    parser.add_argument('--boosted', dest='boosted', type=int, default=True, help='boost planner')
    parser.add_argument('--train_world_model', dest='train_world_model', type=int, default=1,
                        help='train_world_model or not')
    parser.add_argument('--save_model', dest='save_model', type=int, default=1, help='whether to save models')
    parser.add_argument('--user_success_rate_threshold', dest='user_success_rate_threshold', type=float, default=1,
                        help='success rate threshold for user model')
    parser.add_argument('--agent_success_rate_threshold', dest='agent_success_rate_threshold', type=float, default=1,
                        help='success rate threshold for agent model')
    parser.add_argument('--pretrain_discriminator', dest='pretrain_discriminator', type=int, default=0,
                        help='whether to pretrain the discriminator')
    parser.add_argument('--discriminator_nn_type', dest='discriminator_nn_type', type=str, default='RNN',
                        help='NN model structure of the discriminator [MLP, RNN]')
    parser.add_argument('--world_model_nn_type', dest='world_model_nn_type', type=str, default='MLP',
                        help='NN model structure of the discriminator [MLP]')
    parser.add_argument('--train_discriminator', dest='train_discriminator', type=int, default=1,
                        help='whether to train the discriminator')
    parser.add_argument('--model_type', dest='model_type', type=str, default='D3Q', help='model type [DQN, DDQ, D3Q]')
    parser.add_argument('--filter_experience_by_discriminator', dest='filter_experience_by_discriminator', type=int,
                        default=1, help='whether to filter the fake experiences by the discriminator')
    parser.add_argument('--buffer_size_unit', dest='buffer_size_unit', type=int, default=2000,
                        help='the unit of buffer size')
    parser.add_argument('--num_exp_store_per_episode_unit', dest='num_exp_store_per_episode_unit', type=int, default=10,
                        help='the number unit of experience to be stored into agent buffers per episode, which is also the number to be in the real user buffer')
    parser.add_argument('--domain_extension_exp', dest='domain_extension_exp', type=int, default=0,
                        help='whether to do domain extension experiments')
    parser.add_argument('--planning_step_to_buffer', dest='planning_step_to_buffer', type=int, default=1,
                        help='the replay buffer size also follow the planning step concept')
    args = parser.parse_args()
    params = vars(args)
    print('Dialog Parameters: ')
    print(json.dumps(params, indent=2))
# </editor-fold>

# <editor-fold desc="运行参数设置">
dialog_config.run_mode = params['run_mode']  # 0:NL, 1:DA, 2:both
dialog_config.auto_suggest = params['auto_suggest']  # 该参数只与agent_cmd相关

warm_start = params['warm_start']  # 是否启用热启动
warm_start_epochs = params['warm_start_epochs']  # 热启动轮数

success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']  # 运行多少epoch保存一次模型

max_turn = params['max_turn']  # 对话最大轮数
num_episodes = params['episodes']  # epoch数
simulation_epoch_size = params['simulation_epoch_size']  # 验证集大小
batch_size = params['batch_size']  # default = 16
planning_steps = params['planning_steps']

agt = params['agt']  # 智能体id
usr = params['usr']  # 用户（模拟器）id

# 设置随机种子
seed = 2
random.seed(seed)
# </editor-fold>

# <editor-fold desc="加载外部文件">
########## 加载与切分用户目标 ##########
goal_file_path = convertFile(params['goal_file_path'])
all_goal_set = pickle.load(open(goal_file_path, 'rb'))
# 切分用户目标集
split_fold = params.get('split_fold', 5)  # 切分折数，默认为5
goal_set = {'train': [], 'valid': [], 'test': [], 'all': []}
for u_goal_id, u_goal in enumerate(all_goal_set):
    if u_goal_id % split_fold == 1:  # 每5个目标中，有1个分到test集中，4个分到train集中
        goal_set['test'].append(u_goal)
    else:
        goal_set['train'].append(u_goal)
    goal_set['all'].append(u_goal)  # 所有的用户目标都放到all集中

########## 加载电影字典 ##########
dict_path = convertFile(params['dict_path'])
movie_dictionary = pickle.load(open(dict_path, 'rb'))

########## 加载电影数据库 ##########
movie_kb_path = convertFile(params['movie_kb_path'])
movie_kb = pickle.load(open(movie_kb_path, 'rb'), encoding='latin1')

########## 加载动作集与槽集 ##########
act_set = text_to_dict(params['act_set'])  # path to dia act set
slot_set = text_to_dict(params['slot_set'])  # path to slot set
# </editor-fold>

# <editor-fold desc="鉴别器相关">
# 鉴别器参数
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['learning_phase'] = params['learning_phase']
usersim_params['hidden_size'] = params['dqn_hidden_size']

# 实例化鉴别器
discriminator = Discriminator(movie_dict=movie_dictionary, act_set=act_set, slot_set=slot_set, start_set=goal_set,
                              nn_type=params['discriminator_nn_type'], params=usersim_params)


# 使用鉴别器来筛选出高质量的模拟经验
def simulation_epoch_with_gan_control_filter(simulation_epoch_size, use_world_model=True,
                                             filter_experience_by_discriminator=False):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    print("+------------------------+")
    print("| Collecting Experiences |")
    print("+------------------------+")

    res = {}
    for episode in range(simulation_epoch_size):
        # NOTE: dialog_manager.initialize_episode(False) will use world model
        dialog_manager.initialize_episode(not use_world_model)  # 此处使用世界模型
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn(
                filter_experience_by_discriminator=filter_experience_by_discriminator)  # 此处为False，在D3Q函数中为True
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("simulation episode %s: Success" % (episode))
                else:
                    print("simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns) / simulation_epoch_size
    print("simulation success rate %s, ave reward %s, ave turns %s" % (
        res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res


# 预训练鉴别器
def simulation_epoch_for_pretrain_discriminator(simulation_epoch_size):
    for episode in range(simulation_epoch_size):  # 执行simulation_epoch_size次对话
        dialog_manager.initialize_episode(False)  # 使用世界模型
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data=False,
                                                            record_training_data_for_user=False)
    return 0


# </editor-fold>

# <editor-fold desc="智能体设置">
########## Parameters for Agent ##########
agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']
agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['cmd_input_mode'] = params['cmd_input_mode']  # 0 for NL; 1 for dia_act
agent_params['planning_steps'] = params['planning_steps']
agent_params['model_type'] = params['model_type']  # DQN DDQ D3Q
agent_params['buffer_size_unit'] = params['buffer_size_unit']
agent_params['planning_step_to_buffer'] = params['planning_step_to_buffer']

########## Select Agents ##########
if agt == 0:
    agent = AgentCmd(movie_kb, act_set, slot_set, agent_params)
elif agt == 1:
    agent = InformAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 2:
    agent = RequestAllAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 3:
    agent = RandomAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 4:
    agent = EchoAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 5:
    agent = RequestBasicsAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 9:
    agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)
elif agt == 10:
    agent = Agent_NoNStep(movie_kb, act_set, slot_set, agent_params)    # edit different agents here
# add your custom agent here
else:
    pass
# </editor-fold>

# <editor-fold desc="用户（模拟器）设置">
# 用户模拟器参数设置
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['learning_phase'] = params['learning_phase']
usersim_params['hidden_size'] = params['dqn_hidden_size']
usersim_params['world_model_nn_type'] = params['world_model_nn_type']
usersim_params['buffer_size_unit'] = params['buffer_size_unit']

# 选择用户（模拟器）
if usr == 0:  # 真实用户
    user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    user_sim_planning = None
elif usr == 1:  # 基于规则的用户模拟器 & 基于模型的用户模拟器
    user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    user_sim_planning = ModelBasedSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params,
                                            discriminator)
    agent.set_user_planning(user_sim_planning)  # 使用基于模型的用户模拟器进行规划
# add your user simulator here
else:
    pass
# </editor-fold>

# <editor-fold desc="对话管理器设置">
dialog_manager = DialogManager(agent, user_sim, user_sim_planning, act_set, slot_set, movie_kb, discriminator)
# </editor-fold>

# <editor-fold desc="加载预训练NLG和NLU模型">

########## Load trained NLG model ###########
nlg_model_path = convertFile(params['nlg_model_path'])
diaact_nl_pairs = params['diaact_nl_pairs']
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)
if (user_sim_planning != None):
    user_sim_planning.set_nlg_model(nlg_model)

########## Load trained NLU model ##########
nlu_model_path = convertFile(params['nlu_model_path'])
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)
# </editor-fold>

# <editor-fold desc="保存模型与记录">
if not os.path.isdir(params['write_model_dir']):
    os.makedirs(params['write_model_dir'])
with open(os.path.join(params['write_model_dir'], "model_config"), "w+") as f:
    for arg in vars(args):
        f.write("{}: {}\n".format(arg, str(getattr(args, arg))))
    f.close()

########## Save Best Model and Performance Record ##########
best_model = {}
best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0}

performance_records = {}
performance_records['success_rate'] = {}
performance_records['ave_turns'] = {}
performance_records['ave_reward'] = {}
performance_records['use_world_model'] = {}
performance_records['agent_world_model_buffer_size'] = {}
performance_records['agent_user_buffer_size'] = {}
performance_records['discriminator_loss'] = {}
performance_records['world_model_loss'] = {}
performance_records['world_model_buffer_size'] = {}


# save model
def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f.pkl' % (agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    if agt == 10: checkpoint['model'] = copy.deepcopy(agent.dqn.model)
    checkpoint['params'] = params
    try:
        pickle.dump(checkpoint, open(filepath, "w"))
        print('saved model in %s' % (filepath))
    except Exception as e:
        print('Error: Writing model fails: %s' % (filepath))
        print(e)


# Save performance numbers
def save_performance_records(path, agt, records):
    filename = 'agt_%s_performance_records.json' % (agt)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "w"))
        print('saved model in %s' % (filepath))
    except Exception as e:
        print('Error: Writing model fails: %s' % (filepath))
        print(e)


# </editor-fold>

# <editor-fold desc="热启动相关 (by Rule Policy)">
def warm_start_simulation():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_epochs = 0
    for _ in range(1):
        for episode in range(warm_start_epochs):  # 在热启动阶段进行warm_start_epochs次的对话
            dialog_manager.initialize_episode(warm_start=True)  # 此处不启用世界模型
            episode_over = False
            while (not episode_over):  # 与环境交互
                episode_over, reward = dialog_manager.next_turn()
                cumulative_reward += reward
                if episode_over:
                    if reward > 0:
                        successes += 1
                        # if episode == 99:
                        #     print("开始调试：")
                        print("warm_start simulation episode %s: Success" % (episode))
                    else:
                        print("warm_start simulation episode %s: Fail" % (episode))
                    cumulative_turns += dialog_manager.state_tracker.turn_count

            warm_start_run_epochs += 1

        if params['boosted']:  # boost the world model with examples generated by rule agent 是否也为世界模型进行热启动？
            user_sim_planning.train(batch_size, 5)

    # 将回放缓存中的经验存放到pkl文件中
    pickle.dump(agent.experience_replay_pool,
                open('warm_up_experience_pool_seed%d_r%d.pkl' % (seed, successes), 'wb'))
    pickle.dump(agent.experience_replay_pool_from_model,
                open('warm_up_experience_pool_seed%d_r%d_sb.pkl' % (seed, successes), 'wb'))
    pickle.dump(user_sim_planning.training_examples,
                open('warm_up_experience_pool_seed%d_r%d_user.pkl' % (seed, successes), 'wb'))

    # 统计热启动的成功率、平均奖励、平均轮数
    agent.warm_start = 2
    res['success_rate'] = float(successes) / warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward) / warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns) / warm_start_run_epochs
    print("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (
        episode + 1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    # 打印当前回放缓存大小
    print("Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))


# 直接载入预先得到的回放缓存经验
def warm_start_simulation_preload():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_epochs = 0

    agent.experience_replay_pool = pickle.load(open('warm_up_experience_pool_seed4883_r40.pklGod', 'rb'))
    user_sim_planning.training_examples = pickle.load(open('warm_up_experience_pool_seed4883_r40_user.pkl', 'rb'))
    user_sim_planning.train(batch_size, 5)
    agent.warm_start = 2
    print("Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))


# </editor-fold>

# <editor-fold desc="Simulation, DQN, DDQ, D3Q">
# Validation
def simulation_epoch(simulation_epoch_size):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    print("+---------------------+")
    print("|      Validation     |")
    print("+---------------------+")
    res = {}
    for episode in range(simulation_epoch_size):
        dialog_manager.initialize_episode(warm_start=True)
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data=False,
                                                            record_training_data_for_user=False)
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("simulation episode %s: Success" % (episode))
                else:
                    print("simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns) / simulation_epoch_size
    print("simulation success rate %s, ave reward %s, ave turns %s" % (
        res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res


def simulation_dqn():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    simulation_count = 0
    total_simulation_count = 0
    num_real_exp_this_episode = 0
    max_num_real_exp = params['num_exp_store_per_episode_unit'] * (params['planning_steps'] + 1)
    print("+------------------------------+")
    print("| Collecting Experiences (DQN) |")
    print("+------------------------------+")
    res = {}
    while num_real_exp_this_episode < max_num_real_exp:
        # NOTE: dialog_manager.initialize_episode(False) will use world model
        dialog_manager.initialize_episode(True)
        episode_over = False
        record_training_data = True
        simulation_count += 1
        while (not episode_over):
            if num_real_exp_this_episode >= max_num_real_exp:
                record_training_data = False
            else:
                num_real_exp_this_episode += 1

            episode_over, reward = dialog_manager.next_turn(record_training_data=record_training_data)
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("simulation episode {}: Success".format(simulation_count))
                else:
                    print("simulation episode {}: Fail".format(simulation_count))
                cumulative_turns += dialog_manager.state_tracker.turn_count

    total_simulation_count += simulation_count
    res['success_rate'] = float(successes) / total_simulation_count
    res['ave_reward'] = float(cumulative_reward) / total_simulation_count
    res['ave_turns'] = float(cumulative_turns) / total_simulation_count
    print("simulation success rate %s, ave reward %s, ave turns %s" % (
        res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res


def simulation_ddq():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    num_real_exp_this_episode = 0
    num_fake_exp_this_episode = 0
    max_num_real_exp = params['num_exp_store_per_episode_unit']
    max_num_fake_exp = params['num_exp_store_per_episode_unit'] * params['planning_steps']

    print("+-----------------------------+")
    print("| Collecting Experiences (DDQ)|")
    print("+-----------------------------+")
    print("+---------------------------+")
    print("| Collecting Experiences    |")
    print("| From Real Human           |")
    print("+---------------------------+")
    res = {}
    total_simulation_count = 0
    simulation_count = 0
    while num_real_exp_this_episode < max_num_real_exp:
        # NOTE: dialog_manager.initialize_episode(False) will use world model
        dialog_manager.initialize_episode(True)
        episode_over = False
        record_training_data = True
        simulation_count += 1
        while (not episode_over):
            if num_real_exp_this_episode >= max_num_real_exp:
                record_training_data = False
            else:
                num_real_exp_this_episode += 1

            episode_over, reward = dialog_manager.next_turn(
                record_training_data=record_training_data,
                filter_experience_by_discriminator=False)
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("simulation episode {}: Success".format(simulation_count))

                else:
                    print("simulation episode {}: Fail".format(simulation_count))
                cumulative_turns += dialog_manager.state_tracker.turn_count

    print("+---------------------------+")
    print("| Collecting Experiences    |")
    print("| From World Model          |")
    print("+---------------------------+")
    total_simulation_count += simulation_count
    simulation_count = 0
    while num_fake_exp_this_episode < max_num_fake_exp:
        # NOTE: dialog_manager.initialize_episode(False) will use world model
        dialog_manager.initialize_episode(False)
        episode_over = False
        record_training_data = True
        simulation_count += 1
        while (not episode_over):
            if num_fake_exp_this_episode >= max_num_fake_exp:
                record_training_data = False
            else:
                num_fake_exp_this_episode += 1

            episode_over, reward = dialog_manager.next_turn(
                record_training_data=record_training_data,
                filter_experience_by_discriminator=False)

            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print("simulation episode {}: Success".format(simulation_count))
                else:
                    print("simulation episode {}: Fail".format(simulation_count))
                cumulative_turns += dialog_manager.state_tracker.turn_count

    total_simulation_count += simulation_count
    res['success_rate'] = float(successes) / total_simulation_count
    res['ave_reward'] = float(cumulative_reward) / total_simulation_count
    res['ave_turns'] = float(cumulative_turns) / total_simulation_count
    print("simulation success rate %s, ave reward %s, ave turns %s" % (
        res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res


def simulation_d3q():
    res = {}  # 记录D3Q的成功率、平均奖励、平均对话轮数
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    num_real_exp_this_episode = 0
    num_fake_exp_this_episode = 0
    max_num_real_exp = params['num_exp_store_per_episode_unit']
    max_num_fake_exp = params['num_exp_store_per_episode_unit'] * params['planning_steps']  # Planning已融合进这里

    print("+------------------------------+")
    print("| Collecting Experiences (D3Q)|")
    print("+------------------------------+")
    print("+---------------------------+")
    print("| Collecting Experiences    |")
    print("| From Real Human           |")
    print("+---------------------------+")
    simulation_count = 0
    total_simulation_count = 0

    while num_real_exp_this_episode < max_num_real_exp:
        # NOTE: dialog_manager.initialize_episode(False) will use world model
        dialog_manager.initialize_episode(True)  # 该阶段从人类（数据集）中收集真实经验，因此关闭世界模型。
        episode_over = False
        record_training_data = True
        simulation_count += 1
        while (not episode_over):
            if num_real_exp_this_episode >= max_num_real_exp:
                record_training_data = False
            else:
                num_real_exp_this_episode += 1

            episode_over, reward = dialog_manager.next_turn(
                record_training_data=record_training_data,
                filter_experience_by_discriminator=False)  # 此处不启用鉴别器

            cumulative_reward += reward

            if episode_over:
                if reward > 0:
                    successes += 1
                    print("simulation episode {}: Success".format(simulation_count))
                else:
                    print("simulation episode {}: Fail".format(simulation_count))

                cumulative_turns += dialog_manager.state_tracker.turn_count

    print("+---------------------------+")
    print("| Collecting Experiences    |")
    print("| From World Model          |")
    print("+---------------------------+")
    total_simulation_count += simulation_count  # 把上一个阶段的simulation_count汇总进total_simulation_count
    simulation_count = 0  # 重新记录该阶段的simulation_count

    while num_fake_exp_this_episode < max_num_fake_exp:
        dialog_manager.initialize_episode(False)  # 该阶段从世界模型中收集模拟经验，因此启用世界模型
        episode_over = False
        record_training_data = True
        simulation_count += 1

        while (not episode_over):
            if num_fake_exp_this_episode >= max_num_fake_exp:
                record_training_data = False

            episode_over, reward, discriminate_check = dialog_manager.next_turn(
                record_training_data=record_training_data,
                filter_experience_by_discriminator=True)  # 此处开启鉴别器筛选出高质量的模拟经验
            if discriminate_check and record_training_data:
                num_fake_exp_this_episode += 1  # 只有通过鉴别器检验的经验才算。

            cumulative_reward += reward

            if episode_over:
                if reward > 0:
                    successes += 1
                    print("simulation episode {}: Success".format(simulation_count))
                else:
                    print("simulation episode {}: Fail".format(simulation_count))

                cumulative_turns += dialog_manager.state_tracker.turn_count

    total_simulation_count += simulation_count  # total_simulation_count中记录了2个阶段总的simulation_count
    res['success_rate'] = float(successes) / total_simulation_count
    res['ave_reward'] = float(cumulative_reward) / total_simulation_count
    res['ave_turns'] = float(cumulative_turns) / total_simulation_count
    print("simulation success rate %s, ave reward %s, ave turns %s" % (
        res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res  # 返回含统计结果的字典


# </editor-fold>

status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}


def run_episodes(count, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    # 热启动
    if agt == 10 and params['trained_model_path'] == None and warm_start == 1:
        print('warm_start starting ...')
        warm_start_simulation()

        # 把user_actions导出到user_actions.dump文件中
        pickle.dump(dialog_manager.user_actions_for_dump, open('user_actions.dump', 'wb'))
        print('warm_start finished, start RL training ...')

    # 预训练鉴别器
    if agt == 10 and params['pretrain_discriminator']:
        print("pretraining the discriminator...")
        # TODO: use argument
        for _ in range(20):  # 做20次预训练鉴别器——未来可将其参数化
            simulation_epoch_for_pretrain_discriminator(3)  # 使用世界模型进行3次episode的对话
            discriminator_loss = dialog_manager.discriminator.train()  # 训练鉴别器，返回损失
            print("discriminator loss: {}".format(discriminator_loss))

    # num_episode次主循环
    for episode in range(count):
        if params['model_type'] == 'D3Q' and episode == 0:
            simulation_epoch_with_gan_control_filter(3, True)  # 仅在episode == 0时，使用世界模型
        else:
            simulation_epoch_with_gan_control_filter(3, False)  # 不使用世界模型

        # 每轮episode结束后更新目标网络
        # agent.dqn.update_fixed_target_network()

        # 每轮episode结束后更新网络
        agent.dqn.update_network()

        print("Episode: %s" % (episode))  # 打印当前episode
        agent.predict_mode = False
        dialog_manager.initialize_episode(True)  # 不使用世界模型
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data_for_user=False)
            cumulative_reward += reward

            if episode_over:
                if reward > 0:
                    print("Successful Dialog!")
                    successes += 1
                else:
                    print("Failed Dialog!")
                cumulative_turns += dialog_manager.state_tracker.turn_count

        # 实施D3Q、DDQ、DQN算法
        if agt == 10 and params['trained_model_path'] == None:
            agent.predict_mode = True
            user_sim_planning.predict_mode = True

            if params['model_type'] == "D3Q":
                simulation_d3q()
            elif params['model_type'] == "DDQ":
                simulation_ddq()
            elif params['model_type'] == "DQN":
                simulation_dqn()

            # 验证50次
            if episode >= 0:
                agent.predict_mode = False
                user_sim_planning.predict_mode = False
                simulation_res = simulation_epoch(50)
            else:
                simulation_res = {}
                simulation_res['success_rate'] = 0
                simulation_res['ave_turns'] = 42
                simulation_res['ave_reward'] = -60

            # 记录该轮指标
            performance_records['success_rate'][episode] = simulation_res['success_rate']
            performance_records['ave_turns'][episode] = simulation_res['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res['ave_reward']

            # 记录buffer size
            performance_records['agent_user_buffer_size'][episode] = len(agent.experience_replay_pool)
            performance_records['agent_world_model_buffer_size'][episode] = len(agent.experience_replay_pool_from_model)

            # 记录最佳指标
            if simulation_res['success_rate'] > best_res['success_rate']:
                # best_model['model'] = copy.deepcopy(agent)
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['ave_reward'] = simulation_res['ave_reward']
                best_res['ave_turns'] = simulation_res['ave_turns']
                best_res['epoch'] = episode

            # 训练智能体
            user_sim_planning.adversarial = False
            agent.train(batch_size, 1)

            # 训练世界模型
            if params['train_world_model']:
                print("+---------------------+")
                print("|  Train World Model  |")
                print("+---------------------+")

                performance_records['world_model_loss'][episode] = user_sim_planning.train(batch_size, 1)
                performance_records['world_model_buffer_size'][episode] = len(user_sim_planning.training_examples)

            # 训练鉴别器
            if episode > 1 and params['model_type'] == 'D3Q' and params['train_discriminator']:
                print("+---------------------+")
                print("| Train Discriminator |")
                print("+---------------------+")
                discriminator_loss = dialog_manager.discriminator.train()
                performance_records['discriminator_loss'][episode] = discriminator_loss
                print("discriminator loss: {}".format(discriminator_loss))

            agent.predict_mode = False

            print("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (
                performance_records['success_rate'][episode],
                performance_records['ave_reward'][episode],
                performance_records['ave_turns'][episode],
                best_res['success_rate']))

            # 每执行50轮循环，保存模型和performance_records
            path = '{}/duellingdqn.model.epoch{}'.format(params['write_model_dir'], episode)
            if params['save_model'] and episode % 50 == 0:
                agent.save_dqn(path)
            save_performance_records(params['write_model_dir'], agt, performance_records)

        # 打印该次episode的评价指标
        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
            episode + 1,
            count,
            successes,
            episode + 1,
            float(cumulative_reward) / (episode + 1),
            float(cumulative_turns) / (episode + 1)))

    # 打印总循环结束后的评价指标
    print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
        successes,
        count,
        float(cumulative_reward) / count,
        float(cumulative_turns) / count))
    status['successes'] += successes
    status['count'] += count

    # 保存模型和performance_records
    path = '{}/duellingdqn.model.epoch{}'.format(params['write_model_dir'], '_final')  # 保存路径
    if params['save_model']:  # 默认保存模型
        agent.save_dqn(path)
    if agt == 10 and params['trained_model_path'] == None:  # 若agt=10且处于训练模式，追加保存性能记录
        save_performance_records(params['write_model_dir'], agt, performance_records)


run_episodes(num_episodes, status)
