from multiprocessing import set_start_method, Queue

from environment_runner import EnvRunner


def train():
    from agents.double_dqn import Agent
    obs_q = Queue()
    runner = EnvRunner(obs_q)
    agent = Agent(
        runner, obs_q, max_ep_len=1_000,
        # main_updates_per_target_update=100
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
    set_start_method('spawn')

    train()
