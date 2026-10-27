from multiprocessing import Process, Queue


def get_env_runner_fn(constructor):
    def env_runner(in_q, out_q):
        print('whaddup from subproc')
        env = constructor()

        while True:
            obs, _ = env.reset()
            print('waiting for first obs in subproc')
            out_q.put(obs)
            terminal = False
            while True:
                if terminal:
                    out_q.put(obs, 0, False)
                    break

                action = in_q.get()

                # Reached max episode length
                if action is None:
                    break

                obs_next, reward, terminal, _, _ = env.step(action)
                out_q.put((obs, reward, terminal))

                obs = obs_next

    return env_runner


class EnvRunner:
    def __init__(self, env_constructor, out_q):
        self.in_q = Queue()

        env_runner_fn = get_env_runner_fn(env_constructor)

        self.process = Process(
            target=env_runner_fn,
            args=(self.in_q, out_q),
            daemon=True
        )

        self.process.start()
