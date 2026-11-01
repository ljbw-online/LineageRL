from socket import socket
from time import sleep

import cv2
import numpy as np

# Use python -m environments.gta to run this file directly


def get_msg_bytes(msg_num):
    msg = [0] * 17
    msg[0] = msg_num
    return bytes(msg)


reinforcement_resumed_bytes = get_msg_bytes(4)
reinforcement_paused_bytes = get_msg_bytes(5)

# reinforcement_resumed_bytes = bytes(
#     [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# )
# reinforcement_paused_bytes = bytes(
#     [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# )
# position_bytes = bytes([15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# state_bytes = bytes([23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

w_bytes = bytes([22]) + np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
wa_bytes = bytes([22]) + np.array([1, 1, 0, 0], dtype=np.float32).tobytes()
wd_bytes = bytes([22]) + np.array([1, 0, 0, 1], dtype=np.float32).tobytes()
s_bytes = bytes([22]) + np.array([0, 0, 1, 0], dtype=np.float32).tobytes()
sa_bytes = bytes([22]) + np.array([0, 1, 1, 0], dtype=np.float32).tobytes()
sd_bytes = bytes([22]) + np.array([0, 0, 1, 1], dtype=np.float32).tobytes()
a_bytes = bytes([22]) + np.array([0, 1, 0, 0], dtype=np.float32).tobytes()
d_bytes = bytes([22]) + np.array([0, 0, 0, 1], dtype=np.float32).tobytes()
nk_bytes = bytes([22]) + np.array([0, 0, 0, 0], dtype=np.float32).tobytes()


starting_position = np.array([1099, -265, 69], dtype=np.float32)
teleport_bytes = (
    bytes([19, 0, 0, 0]) + starting_position.tobytes() + bytes([0])
)
# teleport_bytes = [0] * 17
# teleport_bytes[0] = 19
# teleport_bytes = bytes(teleport_bytes)

headings = [152, 332]

window_name = 'OpenCV'
action_labels = ['accelerate', 'nothing', 'brake/reverse', 'left', 'right']


def get_heading_bytes(heading):
    return (
        bytes([21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        + np.float32(heading).tobytes()
        + bytes([0])
    )


def read_or_reinit_capture(cap):
    return_val, frame = cap.read()

    while return_val is False:
        print('Reinitialising capture')
        cap.release()
        sleep(1)
        cap = cv2.VideoCapture(2, cv2.CAP_V4L2)  # Capture card
        cap.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return_val, frame = cap.read()

    return frame


def display_captures():
    from visualisation import Window

    window = Window(shape=(720, 1280))

    for i in range(10):
        capture = get_new_capture(index=i)

        window_name = f'Capture {i}'
        return_val = True

        while return_val:
            return_val, frame = capture.read()

            if frame is not None:
                # cv2.imshow(window_name, frame)
                # key_ord = cv2.waitKey(33)
                window.update(frame)
                keys = window.get_keys()
                sleep(0.1)
            else:
                break

            if 'q' in keys:
                window.close()
                break

    # cv2.destroyAllWindows()


def get_new_capture(index=0):
    video_capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    video_capture.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    )
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return video_capture


def render_loop(in_q, out_q, testing):
    # from audio import CallbackPair
    video_capture = get_new_capture()

    # callback_pair = CallbackPair()
    # callback_pair.start()

    while in_q.empty():
        return_val, frame = video_capture.read()

        while return_val is False:
            print('Reinitialising capture')
            video_capture.release()
            sleep(1)
            video_capture = get_new_capture()
            return_val, frame = video_capture.read()

        # It appears that we do need to call waitKey in this subprocess
        # key_ord = cv2.waitKey(waitkey_duration)

        if out_q.empty():
            out_q.put(frame)

        # cv2.imshow(window_name, frame)

    video_capture.release()
    # callback_pair.attempt_to_stop()

    in_q.cancel_join_thread()
    out_q.cancel_join_thread()


class Env:
    def __init__(self, testing=False):
        self.num_actions = 5
        self.max_return = 10_000
        self.random_eval_action = 1
        self.max_steps_per_episode = 1000
        self.evaluation_epsilon = 0.1

        # self.in_q = Queue()
        # self.frame_q = Queue()

        # Start displaying capture card so I can login to Windows if necessary
        # self.rendering_process = Process(
        #     target=render_loop,
        #     args=(self.in_q, self.frame_q, testing),
        #     daemon=True
        # )

        # self.rendering_process.start()

        self.sock = socket()

        print('Connecting socket')
        self.connect_socket()
        print('Socket connected')

        self.timestep_dtype = np.dtype([
            ('observation', np.uint8, (84, 84)),
            ('action', np.int32),
            ('reward', np.float32)
        ])

        self.cap = get_new_capture()

        self.prev_position = None
        self.heading_number = 0
        self.first_step = True

    def connect_socket(self):
        while True:
            try:
                self.sock.connect(("DESKTOP-60DJ6MK", 7001))
                break
            except Exception as exception:
                print(exception)
                sleep(1)

    def reset(self):
        while True:
            try:
                # sets control normals to 0
                self.sock.sendall(reinforcement_resumed_bytes)
                self.sock.sendall(teleport_bytes)
                self.sock.sendall(
                    get_heading_bytes(headings[self.heading_number])
                )
                break
            except Exception as exception:
                print(exception)
                self.sock.close()
                sleep(1)
                self.sock = socket()
                self.connect_socket()

        self.heading_number = (self.heading_number + 1) % len(headings)
        self.first_step = True

        # frame = self.frame_q.get()
        frame = read_or_reinit_capture(self.cap)

        return frame[:, 164:1124]

    def step(self, action):
        terminal = False

        while True:
            try:
                self.send_action(action)
                recv_bytes = self.sock.recv(29)
                break
            except Exception as ex:
                print(ex)
                self.sock.close()
                sleep(1)
                self.sock = socket()
                self.connect_socket()

        position = np.frombuffer(
            recv_bytes, dtype=np.float32, count=3, offset=5
        )

        submerged = bool(recv_bytes[17])

        if self.first_step:
            self.prev_position = position.copy()
            self.first_step = False

        if submerged:
            reward = np.float32(0)
            terminal = True
        else:
            reward = (
                np.sqrt(np.sum((position - self.prev_position) ** 2)) / 100
            )

        self.prev_position = position

        # frame = self.frame_q.get()
        frame = read_or_reinit_capture(self.cap)

        return frame[:, 164:1124], reward, terminal

    def send_action(self, action):
        match action:
            case 0:
                self.sock.sendall(w_bytes)
            case 1:
                self.sock.sendall(nk_bytes)
            case 2:
                self.sock.sendall(s_bytes)
            case 3:
                self.sock.sendall(wa_bytes)
            case 4:
                self.sock.sendall(wd_bytes)
            case _:
                print('Invalid action: {}'.format(action))

    def pause(self):
        self.sock.sendall(reinforcement_resumed_bytes)

    def close(self):
        self.sock.sendall(reinforcement_paused_bytes)
        self.sock.close()
        self.cap.release()

        self.in_q.put(None)
        # Blocks because of SDL
        # self.rendering_process.join()


def test_env():
    episode_reward = 0
    key_ord = 0
    env = Env(stack=True)
    terminated = False
    while True:
        if key_ord == ord('q'):
            break

        episode_reward = 0
        observation = env.reset()
        for step_count in range(1_000):
            if terminated:
                print('terminated')
                terminated = False
                break

            observation = np.hstack(np.split(observation, 4, axis=2))
            cv2.imshow('Observation', observation)

            action = 1  # no-keys

            key_ord = cv2.waitKey(1)

            if key_ord == ord('q'):
                cv2.destroyAllWindows()
                env.close()
                break
            elif key_ord == ord('i'):
                action = 0
            elif key_ord == ord('j'):
                action = 3
            elif key_ord == ord('l'):
                action = 4
            elif key_ord == ord('k'):
                action = 2

            # print('sending action', action)
            # print('key_ord', key_ord)

            next_observation, reward, terminated = env.step(action)
            observation = next_observation

            episode_reward += reward

            print('{:.4f}'.format(reward))

        print(f'episode_reward: {episode_reward}')


if __name__ == '__main__':
    display_captures()
