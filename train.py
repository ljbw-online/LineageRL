from multiprocessing import set_start_method  # , Queue

# from environment_runner import EnvRunner
from environments.gta import Env
# import gymnasium as gym


def get_env():
    return Env()
    # return gym.make('ALE/Breakout-v5')


def train():
    from agents.double_dqn import Agent

    agent = Agent(
        get_env, buffer_len=100_000, max_ep_len=1_000
    )

    num_eval_episodes = 30
    while True:
        agent.train(500_000)

        average_eval_return = agent.evaluate(num_eval_episodes)

        print(f'eval return {average_eval_return:.3}')

        # if average_eval_return > (num_eval_episodes - 1) / num_eval_episodes:
        #     print('env solved')
        #     break


if __name__ == '__main__':
    set_start_method('spawn')

    train()
