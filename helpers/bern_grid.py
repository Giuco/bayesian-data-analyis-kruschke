from typing import Dict, List, Union

import numpy as np
from matplotlib import pyplot as plt

XTICKS = [0, 0.2, 0.4, 0.6, 0.8, 1]


def validate_input(theta: np.array, p_theta: np.array, data: np.array, plot_type: str):
    if np.any((theta > 1) | (theta < 0)):
        raise Exception("Theta values must be between 0 and 1")

    if np.any((p_theta < 0)):
        raise Exception("pTheta must be non-negative")

    if p_theta.sum().round(3) != 1:
        raise Exception("pTheta values must sum to 1.0")

    if not np.all((data == 1) | (data == 0)):
        raise Exception("Data values must be 0 or 1")

    if plot_type not in ["bar", "line"]:
        raise Exception("Only bar and line plot_types are supported")


def set_xticks(ax: plt.Axes):
    ax.xaxis.set_ticks(XTICKS)


def set_multiple_xticks(axs: List[plt.Axes]):
    for ax in axs:
        set_xticks(ax)


def plot_mean_central_tendency(prior_plot: plt.Axes,
                               likelihood_plot: plt.Axes,
                               posterior_plot: plt.Axes,
                               theta: np.array,
                               p_theta: np.array,
                               p_theta_given_data: np.array,
                               p_data_given_theta: np.array):
    prior_mean = np.sum(theta * p_theta)
    prior_plot.text(1.05, np.max([p_theta, p_theta_given_data]) * 0.9, f"Mean = {prior_mean:.3f}",
                    horizontalalignment='right')

    likelihood_mean = np.sum(theta, p_data_given_theta)
    likelihood_plot.text(1.05, p_data_given_theta.max() * 0.9, f"Mean = {likelihood_mean:.3f}",
                         horizontalalignment='right')

    posterior_mean = np.sum(theta * p_theta_given_data)
    posterior_plot.text(1.05, np.max([p_theta, p_theta_given_data]), f"Mean = {posterior_mean:.3f}",
                        horizontalalignment='right')


def plot_mode_central_tendency(prior_plot: plt.Axes,
                               likelihood_plot: plt.Axes,
                               posterior_plot: plt.Axes,
                               theta: np.array,
                               p_theta: np.array,
                               p_theta_given_data: np.array,
                               p_data_given_theta: np.array):
    prior_mode = theta[np.argmax(p_theta)]
    prior_plot.text(1.05, np.max([p_theta, p_theta_given_data]) * 0.9, f"Mode = {prior_mode:.3f}",
                    horizontalalignment='right')

    likelihood_mode = theta[np.argmax(p_data_given_theta)]
    likelihood_plot.text(1.05, p_data_given_theta.max() * 0.9, f"Mode = {likelihood_mode:.3f}",
                         horizontalalignment='right')

    posterior_mode = theta[np.argmax(p_theta_given_data)]
    posterior_plot.text(1.05, np.max([p_theta, p_theta_given_data]) * 0.9, f"Mode = {posterior_mode:.3f}",
                        horizontalalignment='right')


def hdi_of_grid(prob_mass: np.array, cred_mass: float = 0.95) -> Dict[str, Union[np.array, float]]:
    sorted_mass_prob = np.flip(np.sort(prob_mass))
    hdi_height_idx = np.min(np.where(np.cumsum(sorted_mass_prob) >= cred_mass))
    hdi_height = sorted_mass_prob[hdi_height_idx]
    hdi_mass = np.sum(prob_mass[prob_mass >= hdi_height])

    return {
        "indices": np.where(prob_mass >= hdi_height),
        "mass": hdi_mass,
        "height": hdi_height
    }


def plot_hdi_info(prior_plot: plt.Axes, posterior_plot: plt.Axes,
                  p_theta: np.array, p_theta_given_data: np.array, theta: np.array,
                  hdi_mass: float):
    hdi_info = hdi_of_grid(p_theta, cred_mass=hdi_mass)
    height = hdi_info["height"]
    xmin, xmax = theta[hdi_info["indices"]].min(), theta[hdi_info["indices"]].max()
    prior_plot.hlines(y=height, xmin=xmin, xmax=xmax)
    prior_plot.vlines(x=xmin, ymin=0, ymax=height)
    prior_plot.vlines(x=xmax, ymin=0, ymax=height)
    prior_plot.text(xmin, height * 1.3, round(xmin, 3), verticalalignment='bottom', horizontalalignment='center')
    prior_plot.text(xmax, height * 1.3, round(xmax, 3), verticalalignment='bottom', horizontalalignment='center')
    prior_plot.text((xmin + xmax) / 2, prior_plot.get_ylim()[1] * 0.5, f"{hdi_mass * 100:.0f}% HDI",
                    verticalalignment='center', horizontalalignment='center')

    hdi_info = hdi_of_grid(p_theta_given_data, cred_mass=hdi_mass)
    height = hdi_info["height"]
    xmin, xmax = theta[hdi_info["indices"]].min(), theta[hdi_info["indices"]].max()
    posterior_plot.hlines(y=height, xmin=xmin, xmax=xmax)
    posterior_plot.vlines(x=xmin, ymin=0, ymax=height)
    posterior_plot.vlines(x=xmax, ymin=0, ymax=height)
    posterior_plot.text(xmin, height * 1.3, round(xmin, 3), verticalalignment='bottom', horizontalalignment='center')
    posterior_plot.text(xmax, height * 1.3, round(xmax, 3), verticalalignment='bottom', horizontalalignment='center')
    posterior_plot.text((xmin + xmax) / 2, posterior_plot.get_ylim()[1] * 0.5,
                        f"{hdi_mass * 100:.0f}% HDI", verticalalignment='center', horizontalalignment='center')


def plot_bern_grid(theta: np.array,
                   p_theta: np.array,
                   data: np.array,
                   plot_type="bar",
                   color="skyblue",
                   show_central_tendency=None,
                   show_hdi=False,
                   hdi_mass=0.95):
    validate_input(theta, p_theta, data, plot_type)

    z = data.sum()
    n = len(data)

    p_data_given_theta = (theta ** z) * (1 - theta) ** (n - z)
    p_data = np.sum(p_data_given_theta * p_theta)
    p_theta_given_data = (p_data_given_theta * p_theta) / p_data

    f, (prior_plot, likelihood_plot, posterior_plot) = plt.subplots(3, 1)

    getattr(prior_plot, plot_type)(x=theta.round(3), height=p_theta, color=color, width=0.01)
    getattr(likelihood_plot, plot_type)(x=theta.round(3), height=p_data_given_theta, color=color, width=0.01)
    getattr(posterior_plot, plot_type)(x=theta.round(3), height=p_theta_given_data, color=color, width=0.01)

    prior_plot.set_title("Prior")
    prior_plot.set_ylabel("P(θ)")
    prior_plot.set_xlabel("θ")

    likelihood_plot.set_title("Likelihood")
    likelihood_plot.set_ylabel("P(D|θ)")
    likelihood_plot.set_xlabel("θ")

    posterior_plot.set_title("Posterior")
    posterior_plot.set_ylabel("P(θ|D)")
    posterior_plot.set_xlabel("θ")

    likelihood_plot.text(0.1, p_data_given_theta.max() * 0.9, f"Data: z={z}, N={n}", verticalalignment='center',
                         horizontalalignment='center')

    prior_plot.set_ylim((0, 1.1 * np.max([p_theta, p_theta_given_data])))
    likelihood_plot.set_ylim((0, 1.1 * p_data_given_theta.max()))
    posterior_plot.set_ylim((0, 1.1 * np.max([p_theta, p_theta_given_data])))

    [set_xticks(ax) for ax in [prior_plot, likelihood_plot, posterior_plot]]

    if show_central_tendency == "mean":
        plot_mean_central_tendency(prior_plot, likelihood_plot, posterior_plot, theta, p_theta, p_theta_given_data,
                                   p_data_given_theta)
    elif show_central_tendency == "mode":
        plot_mode_central_tendency(prior_plot, likelihood_plot, posterior_plot, theta, p_theta, p_theta_given_data,
                                   p_data_given_theta)

    if show_hdi:
        plot_hdi_info(prior_plot, posterior_plot, p_theta, p_theta_given_data, theta, hdi_mass)

    plt.tight_layout()
    plt.show()
