import torch

from stable_diffusion import get_device


class Options:
    eta_noise_seed_delta: int = 0


opts = Options()
opts.eta_noise_seed_delta = 0
opts.randn_source = "GPU"


def randn(seed, shape, generator=None):
    """Generate a tensor with random numbers from a normal distribution using seed.

    Uses the seed parameter to set the global torch seed; to generate more with that seed, use randn_like/randn_without_seed."""

    manual_seed(seed)

    device = get_device()

    if opts.randn_source == "NV":
        return torch.asarray((generator or nv_rng).randn(shape), device=device)

    if opts.randn_source == "CPU" or device.type == 'mps':
        return torch.randn(shape, device=torch.device("cpu"), generator=generator).to(device)

    return torch.randn(shape, device=device, generator=generator)


def randn_local(seed, shape):
    """Generate a tensor with random numbers from a normal distribution using seed.

    Does not change the global random number generator. You can only generate the seed's first tensor using this function."""
    device = get_device()

    if opts.randn_source == "NV":
        rng = Generator(seed)
        return torch.asarray(rng.randn(shape), device=device)
    device = get_device()
    local_device = torch.device("cpu") if opts.randn_source == "CPU" or device.type == 'mps' else device
    local_generator = torch.Generator(local_device).manual_seed(int(seed))
    return torch.randn(shape, device=local_device, generator=local_generator).to(device)


def randn_like(x):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""

    if opts.randn_source == "NV":
        return torch.asarray(nv_rng.randn(x.shape), device=x.device, dtype=x.dtype)

    if opts.randn_source == "CPU" or x.device.type == 'mps':
        return torch.randn_like(x, device=torch.device("cpu")).to(x.device)

    return torch.randn_like(x)


def randn_without_seed(shape, generator=None):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""
    device = get_device()

    if opts.randn_source == "NV":
        return torch.asarray((generator or nv_rng).randn(shape), device=device)

    if opts.randn_source == "CPU" or device.type == 'mps':
        return torch.randn(shape, device=torch.device("cpu"), generator=generator).to(device)

    return torch.randn(shape, device=device, generator=generator)


def manual_seed(seed):
    """Set up a global random number generator using the specified seed."""

    if opts.randn_source == "NV":
        global nv_rng
        nv_rng = Generator(seed)
        return

    torch.manual_seed(seed)


def create_generator(seed):
    device = get_device()

    if opts.randn_source == "NV":
        return Generator(seed)

    device = torch.device("cpu") if opts.randn_source == "CPU" or device.type == 'mps' else device
    generator = torch.Generator(device).manual_seed(int(seed))
    return generator


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


class ImageRNG:
    def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w

        self.generators = [create_generator(seed) for seed in seeds]

        self.is_first = True

    def first(self):
        noise_shape = self.shape if self.seed_resize_from_h <= 0 or self.seed_resize_from_w <= 0 else (
            self.shape[0], self.seed_resize_from_h // 8, self.seed_resize_from_w // 8)

        xs = []

        for i, (seed, generator) in enumerate(zip(self.seeds, self.generators)):
            subnoise = None
            if self.subseeds is not None and self.subseed_strength != 0:
                subseed = 0 if i >= len(self.subseeds) else self.subseeds[i]
                subnoise = randn(subseed, noise_shape)

            if noise_shape != self.shape:
                noise = randn(seed, noise_shape)
            else:
                noise = randn(seed, self.shape, generator=generator)

            if subnoise is not None:
                noise = slerp(self.subseed_strength, noise, subnoise)

            if noise_shape != self.shape:
                x = randn(seed, self.shape, generator=generator)
                dx = (self.shape[2] - noise_shape[2]) // 2
                dy = (self.shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty:ty + h, tx:tx + w] = noise[:, dy:dy + h, dx:dx + w]
                noise = x

            xs.append(noise)

        eta_noise_seed_delta = opts.eta_noise_seed_delta or 0
        if eta_noise_seed_delta:
            self.generators = [create_generator(seed + eta_noise_seed_delta) for seed in self.seeds]

        return torch.stack(xs).to(get_device())

    def next(self):
        if self.is_first:
            self.is_first = False
            return self.first()

        xs = []
        for generator in self.generators:
            x = randn_without_seed(self.shape, generator=generator)
            xs.append(x)

        return torch.stack(xs).to(get_device())


"""RNG imitiating torch cuda randn on CPU. You are welcome.

Usage:

```
g = Generator(seed=0)
print(g.randn(shape=(3, 4)))
```

Expected output:
```
[[-0.92466259 -0.42534415 -2.6438457   0.14518388]
 [-0.12086647 -0.57972564 -0.62285122 -0.32838709]
 [-1.07454231 -0.36314407 -1.67105067  2.26550497]]
```
"""

import numpy as np

philox_m = [0xD2511F53, 0xCD9E8D57]
philox_w = [0x9E3779B9, 0xBB67AE85]

two_pow32_inv = np.array([2.3283064e-10], dtype=np.float32)
two_pow32_inv_2pi = np.array([2.3283064e-10 * 6.2831855], dtype=np.float32)


def uint32(x):
    """Converts (N,) np.uint64 array into (2, N) np.unit32 array."""
    return x.view(np.uint32).reshape(-1, 2).transpose(1, 0)


def philox4_round(counter, key):
    """A single round of the Philox 4x32 random number generator."""

    v1 = uint32(counter[0].astype(np.uint64) * philox_m[0])
    v2 = uint32(counter[2].astype(np.uint64) * philox_m[1])

    counter[0] = v2[1] ^ counter[1] ^ key[0]
    counter[1] = v2[0]
    counter[2] = v1[1] ^ counter[3] ^ key[1]
    counter[3] = v1[0]


def philox4_32(counter, key, rounds=10):
    """Generates 32-bit random numbers using the Philox 4x32 random number generator.

    Parameters:
        counter (numpy.ndarray): A 4xN array of 32-bit integers representing the counter values (offset into generation).
        key (numpy.ndarray): A 2xN array of 32-bit integers representing the key values (seed).
        rounds (int): The number of rounds to perform.

    Returns:
        numpy.ndarray: A 4xN array of 32-bit integers containing the generated random numbers.
    """

    for _ in range(rounds - 1):
        philox4_round(counter, key)

        key[0] = key[0] + philox_w[0]
        key[1] = key[1] + philox_w[1]

    philox4_round(counter, key)
    return counter


def box_muller(x, y):
    """Returns just the first out of two numbers generated by Box–Muller transform algorithm."""
    u = x * two_pow32_inv + two_pow32_inv / 2
    v = y * two_pow32_inv_2pi + two_pow32_inv_2pi / 2

    s = np.sqrt(-2.0 * np.log(u))

    r1 = s * np.sin(v)
    return r1.astype(np.float32)


class Generator:
    """RNG that produces same outputs as torch.randn(..., device='cuda') on CPU"""

    def __init__(self, seed):
        self.seed = seed
        self.offset = 0

    def randn(self, shape):
        """Generate a sequence of n standard normal random variables using the Philox 4x32 random number generator and the Box-Muller transform."""

        n = 1
        for x in shape:
            n *= x

        counter = np.zeros((4, n), dtype=np.uint32)
        counter[0] = self.offset
        counter[2] = np.arange(n,
                               dtype=np.uint32)  # up to 2^32 numbers can be generated - if you want more you'd need to spill into counter[3]
        self.offset += 1

        key = np.empty(n, dtype=np.uint64)
        key.fill(self.seed)
        key = uint32(key)

        g = philox4_32(counter, key)

        return box_muller(g[0], g[1]).reshape(shape)  # discard g[2] and g[3]
