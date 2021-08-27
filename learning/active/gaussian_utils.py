import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

def approximate_mixture_with_gaussian(mus, sigmas):
    mu = np.mean(mus, axis=0)
    sigma = np.mean(sigmas + mus**2, axis=0) - mu**2

    return mu, sigma


def entropy_of_gaussian(mu, sigma):
    return 0.5 + 0.5 * np.log(2 * np.pi * sigma)

def entropy_of_categorical(c):
    return ((c + 1e-6) * np.log((c + 1e-6))).sum()

def sample_from_mixture(mus, sigmas, size=1):
    N = mus.shape[0]
    i = np.random.randint(low=0, high=N, size=size)
    s = np.random.randn(size)
    return mus[i] + s * np.sqrt(sigmas[i])

def mc_approximate_entropy_of_mixture(mus, sigmas, n_samples):
    s = sample_from_mixture(mus, sigmas, n_samples) 
    w, bin_edges = np.histogram(s, density=True, bins=20)
    # plt.plot(bin_edges[:-1], w)
    # plt.show()
    return entropy_of_categorical(w)

def gaussian_approximate_entropy_of_mixture(mus, sigmas):
    mu, sigma = approximate_mixture_with_gaussian(mus, sigmas)
    return entropy_of_gaussian(mu, sigma)

def get_random_mixture(N):
    mus = np.random.uniform(low=-2, high=2, size=N)
    sigmas = np.random.uniform(low=0.1, high=2, size=N)
    return mus, sigmas

def plot_mixture_vs_gaussian(mus, sigmas):
    N = mus.shape[0]
    x = np.linspace(-8, 8, 1000)
    y = np.zeros_like(x)
    for m, s in zip(mus, sigmas):
        # plt.plot(x, norm.pdf(x, m, s), c='b')
        y += norm.pdf(x, m, s)

    plt.plot(x, y/N)

    mu, sigma = approximate_mixture_with_gaussian(mus, sigmas)
    plt.plot(x, norm.pdf(x, mu, sigma))

    plt.show()

def plot_mixture_entropy_appxoimation():
    mus, sigmas = get_random_mixture(2)
    plot_mixture_vs_gaussian(mus, sigmas)
    ents = []
    xs = np.arange(10, 2000)
    for x in xs:
        ents.append(mc_approximate_entropy_of_mixture(mus, sigmas, x))

    plt.plot(xs, ents)
    plt.plot(xs, np.ones_like(xs) * gaussian_approximate_entropy_of_mixture(mus, sigmas))
    plt.show()


if __name__ == '__main__':

    # plot_mixture_vs_gaussian(3)
    plot_mixture_entropy_appxoimation()