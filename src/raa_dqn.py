import torch
import sys
import gym.spaces
import itertools
import numpy as np
import random
import threading
from utils.replay_buffer import *
from utils.schedules import *
from utils.gym_setup import *
#from src.logger import Logger
from src.anderson_alpha import RAA

from scipy.optimize import brentq
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def dqn_learning(env,
                 omega,
                 q_func,
                 optimizer_spec,
                 exploration=LinearSchedule(1000000, 0.1),
                 max_steps=20e6,
                 replay_buffer_size=1000000,
                 batch_size=32,
                 sample_size=128,
                 gamma=0.99,
                 beta=0.05,
                 reg_scale=0.1,
                 use_restart=True,
                 learning_starts=50000,
                 learning_freq=4,
                 frame_history_len=4,
                 target_update_freq=2000,
                 save_path=None,
                 AA=0,
                 soft=0):
    """Run Deep Q-learning algorithm with regularized anderson acceleration.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    max_steps: float
        Maximal steps.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    # Set the logger
    #logger = Logger(save_path)

    ###############
    # BUILD MODEL #
    ###############

    start_time = time.time()

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
        in_channels = input_shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
        in_channels = input_shape[2]
    num_actions = env.action_space.n

    # define Q target and Q
    Q = q_func(in_channels, num_actions).to(device)
    Q_targets = []
    MAX_NUM = 5
    for i in range(MAX_NUM):
        Q_targets.append(q_func(in_channels, num_actions).to(device))

    # initialize anderson
    anderson = RAA(MAX_NUM, use_restart, reg_scale)

    # initialize optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ######

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    SAVE_MODEL_EVERY_N_STEPS = 100000
    saved_scalars = []
    stop = False
    restart = True
    cur_num = 1
    clipped_error = torch.FloatTensor([0]).to(device)


    for t in itertools.count():
        # 1. Step the env and store the transition
        # store last frame, returned idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get observations to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()  # torch

        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs = observations.unsqueeze(0) / 255.0
                with torch.no_grad():
                    q_value_all_actions = Q(obs)
                action = (q_value_all_actions.data.max(1)[1])[0]
            else:
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]

        obs, reward, done, info = env.step(action)

        # clipping the reward, noted in nature paper
        reward = np.clip(reward, -1.0, 1.0)

        # store effect of action
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        # reset env if reached episode boundary
        if done:
            obs = env.reset()

        # update last_obs
        last_obs = obs

        # 2. Perform experience replay and train the network.
        # if the replay buffer contains enough samples...
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(sample_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(sample_size)
            obs_t = obs_t / 255.0
            act_t = torch.LongTensor(act_t).to(device)
            rew_t = torch.FloatTensor(rew_t).to(device)
            obs_tp1 = obs_tp1 / 255.0
            done_mask = done_mask

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            q_values = Q(obs_t[:batch_size, :])
            q_s_a = q_values.gather(1, act_t[:batch_size].unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            if restart:
                cur_num = 1
                restart = False

                # get the Q values for best actions in obs_tp1
                # based off frozen Q network
                # max(Q(s', a', theta_i_frozen)) wrt a'
                q_tp1_values = Q_targets[-1](obs_tp1[:batch_size, :]).detach()

                #max operator as default.
                q_s_a_prime, _ = q_tp1_values.max(1)

                ################# soft ######################
                if soft == 0:  #
                    q_s_a_prime, _ = q_tp1_values.max(1)
                elif soft == 1:  # mellowmax
                    #c = q_tp1_values.max()

                    #q_s_a_prime = c + torch.log(torch.sum(torch.exp(omega * (q_tp1_values - c)), 1) / num_actions) / omega
                    c = torch.max(q_tp1_values[0])
                    q_s_a_prime = (torch.logsumexp(omega * (q_tp1_values - c), 1) - np.log(num_actions)) / omega + c

                else:  # softmax
                    q_s_a_prime = torch.softmax(omega * q_tp1_values, dim=1)
                    q_s_a_prime = q_s_a_prime.mul(q_tp1_values)
                    q_s_a_prime = torch.sum(q_s_a_prime, dim=1)

                # if current state is end of episode, then there is no next Q value
                q_rhs = rew_t[:batch_size] + gamma * (1 - done_mask[:batch_size]) * q_s_a_prime
            else:
                cur_num += 1
                num = min(MAX_NUM, cur_num)

                cat_obs = torch.cat((obs_t, obs_tp1), 0)
                
                qs_target_t_aa, qs_target_tp1_aa = [], []
                for i in range(num, 0, -1):
                    q_target = Q_targets[-i](cat_obs).detach()
                
                    q_aa = q_target[:sample_size, :].gather(1, act_t.unsqueeze(1))
                    qs_target_t_aa.append(q_aa.t())

                    # max operator as default.
                    q_next_aa, _ = q_target[sample_size:, :].max(1)

                    ################# soft ######################
                    if soft == 0:  # hard
                        q_next_aa, _ = q_target[sample_size:, :].max(1)  # obtain Q(s_t+1, max a)
                    elif soft == 1:  # mellowmax
                        #c = q_target[sample_size:, :].max()
                        #q_next_aa = c + torch.log(
                        #    torch.sum(torch.exp(omega * (q_target[sample_size:, :] - c)), 1) / num_actions) / omega
                        c = torch.max(q_target[sample_size:, :][0])
                        q_next_aa = (torch.logsumexp(omega * (q_target[sample_size:, :] - c), 1) - np.log(num_actions)) / omega + c


                    else:  # softmax
                        q_next_aa = torch.softmax(omega * q_target[sample_size:, :], dim=1)
                        q_next_aa = q_next_aa.mul(q_target[sample_size:, :])  # element-wise
                        q_next_aa = torch.sum(q_next_aa, dim=1)

                    qs_target_tp1_aa.append(q_next_aa.unsqueeze(0))

                qs_target_t_values = torch.cat(qs_target_t_aa, 0)
                qs_target_tp1_values = torch.cat(qs_target_tp1_aa, 0)

           
                F_qs_target_t = torch.cat([(rew_t + gamma * (1 - done_mask) * q).unsqueeze(0)
                                           for q in qs_target_tp1_values], 0)

                alpha = 0
                restart = False

                # (5) important 5: compute the optimal alpha by function anderson
                if AA == 0:  # vanilla AA
                    alpha, restart = anderson.calculate(qs_target_t_values, F_qs_target_t)
                else:  # AA == 1: # new regularization
                    alpha, restart = anderson.calculate_newReg(qs_target_t_values, F_qs_target_t)

                # get Q values from frozen network for next state and chosen action
                # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
                #hybird_qs_target_tp1 = beta * qs_target_t_values[:, :batch_size] + \
                #                       (1 - beta) * F_qs_target_t[:, :batch_size]
                #q_rhs = (hybird_qs_target_tp1.t().mm(alpha)).detach()
                #q_rhs = q_rhs.squeeze(1)

                aa_q = qs_target_t_values[:, :batch_size].t().mm(alpha).detach()
                aa_Tq = F_qs_target_t[:, :batch_size].t().mm(alpha).detach()

                q_rhs = beta * aa_q + (1 - beta) * aa_Tq
                q_rhs = q_rhs.squeeze(1)

            # Compute Bellman error
            # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
            error = q_rhs - q_s_a

            # clip the error and flip
            clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()
            q_s_a.backward(clipped_error.data)

            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_targets[0].load_state_dict(Q.state_dict())
                Q_targets.append(Q_targets[0])
                Q_targets.remove(Q_targets[0])

        # 3. Log progress
        #if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            #if save_path is not None:
                #torch.save(Q.state_dict(), '%s/net.pth' % save_path)

        if t % LOG_EVERY_N_STEPS == 0:
            underlying_env = get_wrapper_by_name(env, "Monitor")
            internal_steps = underlying_env.get_total_steps()
            stop = (internal_steps >= max_steps)
            episode_rewards = underlying_env.get_episode_rewards()
            num_episode = len(episode_rewards)

            if num_episode > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                saved_scalars.append([t, internal_steps, num_episode, mean_episode_reward,
                                      clipped_error.mean().data.cpu().numpy()])
                np.save('%s/scalars.npy' % save_path, saved_scalars)

            end_time = time.time()
            print("---------------------------------")
            print("Wrapped - Atari (steps) %d-%d" % (t, internal_steps))
            print("episodes %d" % num_episode)
            print("mean episode reward %f" % mean_episode_reward)
            print("best mean episode reward %f" % best_mean_episode_reward)
            print("exploration %f" % exploration.value(t))
            print('last time: ' + (str(end_time-start_time)))
            start_time = end_time

            sys.stdout.flush()

            # ============ TensorBoard logging ============#
            info = {'num_episodes': len(episode_rewards),
                    'exploration': exploration.value(t),
                    'mean_episode_reward_last_100': mean_episode_reward
                    }
            #for tag, value in info.items():
            #    logger.scalar_summary(tag, value, t + 1)

        # 4. Check the stop criteria
        if stop:
            break
