from environments.follow_the_dots import Env
from agents.double_dqn import Agent


def get_env():
    return Env(4)


def train():
    agent = Agent(
        get_env, buffer_len=32_000, max_ep_len=100,
        main_updates_per_target_update=100
    )

    num_eval_episodes = 40
    while True:
        agent.train(10_000)

        average_eval_return = agent.evaluate(num_eval_episodes)

        print(f'eval return {average_eval_return:.3}')

        if average_eval_return > (num_eval_episodes - 1) / num_eval_episodes:
            print('env solved')
            break


if __name__ == '__main__':
    train()
