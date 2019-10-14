import argparse
import datetime
import os
import sys

import gym
import numpy as np
import itertools
import torch
from box import Box

from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory

from loguru import logger as log

DIR = os.path.dirname(os.path.realpath(__file__))


@log.catch
def main():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--policy', default="Gaussian",
                        help='algorithm to use: Gaussian | Deterministic')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default:True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False,
                        metavar='G',
                        help='Temperature parameter α automaically adjusted.')
    parser.add_argument('--seed', type=int, default=456, metavar='N',
                        help='random seed (default: 456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=2000001, metavar='N',
                        help='maximum number of steps (default: 2000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1,
                        metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--resume-name', default=None,
                        help='Name of saved model to load')
    args, unknown = parser.parse_known_args()

    # Import custom envs
    import gym_match_input_continuous
    import deepdrive_2d

    # TesnorboardX
    run_name = '{}_SAC_{}_{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.env_name,
        args.policy,
        "autotune" if args.automatic_entropy_tuning else "")

    # Log to file
    os.makedirs('logs', exist_ok=True)
    log.add(f'logs/{run_name}.log')

    log.info(' '.join(sys.argv))

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.reset()

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    if args.resume_name:
        agent.load_model(f'{DIR}/models/sac_actor_runs/{args.resume_name}',
                         f'{DIR}/models/sac_critic_runs/{args.resume_name}')



    run_name = 'runs/' + run_name

    writer = SummaryWriter(logdir=run_name)

    # Memory
    memory = ReplayMemory(args.replay_size)

    train(agent, args, env, memory, run_name, writer)
    env.close()


def train(agent, args, env, memory, run_name, writer):
    total_numsteps = 0
    updates = 0
    run_eval = get_run_eval()
    wins = 0
    for i_episode in itertools.count(1):

        episode_reward, episode_steps, total_numsteps, updates, info = \
            run_episode(agent=agent, args=args, env=env, memory=memory,
                        total_numsteps=total_numsteps, updates=updates,
                        writer=writer)

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        info = dbox(info)
        if 'all_time' in info.tfx:
            for name, value in info.tfx.all_time.items():
                writer.add_scalar(f'all_time_train/{name}', value, i_episode)
                if name == 'won':
                    wins += 1
                    writer.add_scalar(f'all_time_train/win_pct',
                                      100 * wins / i_episode, i_episode)
        log.info(
            "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".
            format(i_episode, total_numsteps, episode_steps,
                   round(episode_reward, 2)))

        if i_episode % 10 == 0 and args.eval == True:
            run_eval(agent, env, i_episode, run_name, writer)


def get_run_eval():
    closure = Box(total_episode_steps=0, wins=0, eval_episodes=0)

    def run_eval(agent, env, i_episode, run_name, writer):
        total_reward = 0
        episodes = 10
        for _ in range(episodes):
            closure.eval_episodes += 1
            state = env.reset()
            episode_reward = 0
            done = False
            info = Box(default_box=True)
            while not done:
                action = agent.select_action(state, eval=True)

                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                if 'tfx' in info:
                    tfx_stats = info['tfx']
                    for name, value in tfx_stats.items():
                        if name != 'all_time':
                            writer.add_scalar(f'info_test/{name}', value,
                                              closure.total_episode_steps)

                state = next_state
                closure['total_episode_steps'] += 1
            info = dbox(info)
            if 'all_time' in info.tfx:
                for name, value in info.tfx.all_time.items():
                    writer.add_scalar(f'all_time_eval/{name}', value,
                                      closure.eval_episodes)
                    if name == 'won':
                        closure.wins += 1
                        writer.add_scalar(
                            f'all_time_eval/win_pct',
                            100 * closure.wins / closure.eval_episodes,
                            closure.eval_episodes)

            total_reward += episode_reward
        avg_reward = total_reward / episodes

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        log.info("----------------------------------------")
        log.info("Test Episodes: {}, Avg. Reward: {}".format(
            episodes, round(avg_reward, 2)))
        log.info("----------------------------------------")

        if i_episode % 100 == 0:
            log.info('Saving model...')
            agent.save_model(run_name)
            log.info('Done saving model')
    return run_eval


def run_episode(agent, args, env, memory, total_numsteps, updates, writer):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    info = Box(default_box=True)
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                    agent.update_parameters(memory, args.batch_size, updates)
                if updates % 100 == 0:
                    write_update_stats(alpha, critic_1_loss, critic_2_loss,
                                       ent_loss, policy_loss, updates, writer)
                updates += 1

        next_state, reward, done, info = env.step(action)  # Step

        if 'tfx' in info and (done or total_numsteps % 100 == 0):
            tfx_stats = info['tfx']
            for name, value in tfx_stats.items():
                if name != 'all_time':
                    writer.add_scalar(f'info_train/{name}', value,
                                      total_numsteps)

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        # Append transition to memory
        memory.push(state, action, reward, next_state, mask)

        state = next_state


    return episode_reward, episode_steps, total_numsteps, updates, info


def write_update_stats(alpha, critic_1_loss, critic_2_loss, ent_loss,
                       policy_loss, updates, writer):
    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
    writer.add_scalar('loss/policy', policy_loss, updates)
    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
    writer.add_scalar('entropy_temprature/alpha', alpha, updates)


def dbox(obj=None, **kwargs):
    if kwargs:
        obj = dict(kwargs)
    else:
        obj = obj or {}
    return Box(obj, default_box=True)


if __name__ == '__main__':
    main()

