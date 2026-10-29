import jax.numpy as jnp
from flax import nnx
# import jax


class MLP(nnx.Module):
    def __init__(self, obs_shape, num_actions, rngs):
        self.size = obs_shape[0] * obs_shape[1]
        self.linear0 = nnx.Linear(self.size, 128, rngs=rngs)
        self.linear1 = nnx.Linear(128, num_actions, rngs=rngs)

    def __call__(self, obs_batch):
        obs_batch = obs_batch.reshape(obs_batch.shape[0], -1) / 255.0
        a = self.linear0(obs_batch)
        a = nnx.gelu(a)
        a = self.linear1(a)
        return a


class CNN(nnx.Module):
    def __init__(self, obs_shape, num_actions, rngs):
        _, width, channels = obs_shape
        conv2_out_width = int(jnp.ceil(width / 8))
        linear0_in_size = (conv2_out_width ** 2) * 64

        self.conv0 = nnx.Conv(
            channels, 32, kernel_size=(8, 8), strides=(4, 4), rngs=rngs
        )

        self.conv1 = nnx.Conv(
            32, 64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs
        )

        self.conv2 = nnx.Conv(
            64, 64, kernel_size=(3, 3), strides=(1, 1), rngs=rngs
        )

        self.linear0 = nnx.Linear(linear0_in_size, 1024, rngs=rngs)
        self.linear1 = nnx.Linear(1024, num_actions, rngs=rngs)

    def __call__(self, x):
        x = x / 255.0

        x = self.conv0(x)
        x = nnx.gelu(x)

        x = self.conv1(x)
        x = nnx.gelu(x)

        x = self.conv2(x)
        x = nnx.gelu(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        x = self.linear0(x)
        x = nnx.gelu(x)

        x = self.linear1(x)

        return x


if __name__ == '__main__':
    cnn = CNN((84, 84, 4), 4, nnx.Rngs(57))

    output = cnn(jnp.ones((1, 84, 84, 4), dtype=jnp.uint8))

    print(output[0])
