""" Tools and experiments in service of approximating the entropy of a
gaussian mixture pdf """
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

# NOTE(izzy): throughout this, the variable sigma actually refers to the
# quantity sigma^2. i just didn't want to have to use a bunch of squares
# and square roots everywhere

############################## Closed Form ##############################
def approximate_mixture_with_gaussian(mus, sigmas):
    mu = np.mean(mus, axis=0)
    sigma = np.mean(sigmas + mus**2, axis=0) - mu**2
    return mu, sigma

def product_of_gaussians(mus, sigmas):
    inv_sigmas = 1./sigmas
    sigma = 1. / np.sum(inv_sigmas, axis=0)
    mu = sigma * np.sum(inv_sigmas * mus, axis=0)
    return mu, sigma

def entropy_of_gaussian(mu, sigma):
    return 0.5 + 0.5 * np.log(2 * np.pi * sigma)

def entropy_of_categorical(c):
    return ((c + 1e-6) * np.log((c + 1e-6))).sum()

############################## Monte Carlo ##############################
def sample_from_mixture(mus, sigmas, size=1):
    N = mus.shape[0]
    i = np.random.randint(low=0, high=N, size=size)
    s = np.random.randn(size)
    return mus[i] + s * np.sqrt(sigmas[i])

def mc_entropy(mus, sigmas, n_samples):
    # draw samples
    x = sample_from_mixture(mus, sigmas, n_samples)
    # compute the likelihood under the mixture pdf
    y = np.zeros_like(x)
    for m, s in zip(mus, sigmas):
        y += norm.pdf(x, m, s)
    y /= mus.shape[0]
    # compute entropy
    return -np.log(y).mean()

def get_random_mixture(N):
    mus = np.random.uniform(low=-3, high=3, size=N)
    sigmas = np.random.uniform(low=0.1, high=2, size=N)
    return mus, sigmas

############################## Plotting ##############################
def plot_mixture(mus, sigmas, ax=plt.gca()):
    N = mus.shape[0]
    x = np.linspace(-8, 8, 1000)
    y = np.zeros_like(x)
    for m, s in zip(mus, sigmas):
        y += norm.pdf(x, m, s)

    ax.plot(x, y/N, label='Mixture')

def plot_mixture_vs_gaussian(mus, sigmas, ax=plt.gca()):
    plot_mixture(mus, sigmas, ax)
    x = np.linspace(-8, 8, 1000)

    mu, sigma = approximate_mixture_with_gaussian(mus, sigmas)
    ax.plot(x, norm.pdf(x, mu, sigma), label='Moment Match')
    mu, sigma = product_of_gaussians(mus, sigmas)
    ax.plot(x, norm.pdf(x, mu, sigma), label='Product')
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.set_title('pdf')
    ax.legend()


def plot_mixture_entropy_appxoimation(mus, sigmas, ax=plt.gca()):
    ents = []
    xs = np.arange(10, 5000)
    for x in xs:
        ents.append(mc_entropy(mus, sigmas, x))

    ax.plot(xs, ents, label='Monte Carlo')
    ax.plot(xs, np.ones_like(xs) * entropy_of_gaussian(*approximate_mixture_with_gaussian(mus, sigmas)), label='Moment Match')
    ax.plot(xs, np.ones_like(xs) * entropy_of_gaussian(*product_of_gaussians(mus, sigmas)), label='Product')
    ax.set_xlabel('Num samples')
    ax.set_ylabel('H(x)')
    ax.set_title('Entropy')
    ax.legend()

############################## Main ##############################
if __name__ == '__main__':
    mus, sigmas = get_random_mixture(4)
    # s = sample_from_mixture(mus, sigmas, 100000)
    # plt.hist(s, density=True)
    # plot_mixture(mus, sigmas)
    # plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot_mixture_vs_gaussian(mus, sigmas, ax=axes[0])
    plot_mixture_entropy_appxoimation(mus, sigmas, ax=axes[1])
    plt.show()
