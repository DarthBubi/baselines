import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_sum

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


def scale_action(action, min_action, max_action):
    return action * max_action * (action >= 0) + action * abs(min_action) * (action < 0)


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, logdir=None, load_policy=False):
    rank = MPI.COMM_WORLD.Get_rank()

    min_action = env.action_space.low
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)

        if load_policy:
            U.load_state(os.path.join(logdir, "model-50"))
            agent.memory.load(os.path.join(logdir, "memory_pickle.pkl"))
            logger.info("Loaded " + str(agent.memory.nb_entries) + " from saved memory.")

        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        epoch_episode_ball_hits = []
        epoch_episode_target_hits = []

        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(scale_action(action, min_action, max_action))  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        epoch_episode_ball_hits.append(int(info['ball_hit']))
                        epoch_episode_target_hits.append(int(info['target_hit']))

                        if info['ball_hit']:
                            pass
                        if info['target_hit']:
                            pass

                        agent.reset()
                        obs = env.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(scale_action(action, min_action, max_action))  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()

            # combined_stats = stats.copy()
            # combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            # combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            # combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            # combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            # combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            # combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            # combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            # combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            # combined_stats['total/duration'] = duration
            # combined_stats['total/steps_per_second'] = float(t) / float(duration)
            # combined_stats['total/episodes'] = episodes
            # combined_stats['rollout/episodes'] = epoch_episodes
            # combined_stats['rollout/actions_std'] = np.std(epoch_actions)

            # Evaluation statistics.
            # if eval_env is not None:
            #     combined_stats['eval/return'] = eval_episode_rewards
            #     combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            #     combined_stats['eval/Q'] = eval_qs
            #     combined_stats['eval/episodes'] = len(eval_episode_rewards)

            # def as_scalar(x):
            #     if isinstance(x, np.ndarray):
            #         assert x.size == 1
            #         return x[0]
            #     elif np.isscalar(x):
            #         return x
            #     else:
            #         raise ValueError('expected scalar, got %s' % x)
            #
            # combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            # combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            # combined_stats['total/epochs'] = epoch + 1
            # combined_stats['total/steps'] = t

            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            # Rollout statistics.
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)
            combined_stats['rollout/ball_hits'] = mpi_mean(epoch_episode_ball_hits)
            combined_stats['rollout/target_hits'] = mpi_mean(epoch_episode_target_hits)

            # Train statistics.
            combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

            # Total statistics.
            combined_stats['total/duration'] = mpi_mean(duration)
            combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
            combined_stats['total/episodes'] = mpi_mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = mpi_mean(np.mean(eval_episode_rewards_history))
                combined_stats['eval/Q'] = mpi_mean(eval_qs)
                combined_stats['eval/episodes'] = mpi_mean(len(eval_episode_reward))

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

                saver.save(sess, os.path.join(logdir, "model"), global_step=epoch)
                agent.memory.save(os.path.join(logdir, 'memory_pickle.pkl'))


def test(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
         normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
         popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
         tau=0.01, eval_env=None, param_noise_adaption_interval=50, logdir=None, load_policy=False):
    rank = MPI.COMM_WORLD.Get_rank()

    min_action = env.action_space.low
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    if render:
        env.render()

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        saver = tf.train.Saver()
        if logdir:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            # saver.restore(sess, tf.train.latest_checkpoint("/home/johannes/Documents/Doggy/doggyPC/HeRoStack/logs/openai-2018-05-21-12-30-19-100857"))
            # saver.restore(sess, tf.train.latest_checkpoint("/home/johannes/Documents/Doggy/doggyPC/HeRoStack/logs/openai-2018-05-20-13-26-41-439177"))
            # saver.restore(sess, tf.train.latest_checkpoint("/home/johannes/Documents/Doggy/doggyPC/HeRoStack/logs/openai-2018-05-22-12-28-40-462267"))
            saver.restore(sess, tf.train.latest_checkpoint("/home/johannes/Documents/Doggy/doggyPC/HeRoStack/logs/openai-2018-05-23-18-13-46-195987"))

        for _ in range(10):
            obs = env.reset()
            done = False
            episode_r = 0
            while not done:
                action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
                obs, r, done, info = env.step(scale_action(action, min_action, max_action))
                episode_r += r
            print('Episode Reward ', episode_r)
