import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns
import pandas as pd
import wandb
from scipy.interpolate import interp1d
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, NamedSharding

from configs import all_configs as all_configs
from network import GRU_Opt as optim
from utils import ParameterReshaper

api = wandb.Api()


labs = [""]


def set_size(width=487.8225, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    golden_ratio = 1.2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    subplots = (1, 1)
    fig_height_in = fig_width_in / golden_ratio * (subplots[0] / subplots[1])
    # Figure height in inches
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


episode_lengths = {
    "asterix": 1000,
    "pendulum": 200,
    "acrobot": 500,
    "cartpole": 500,
    "spaceinvaders": 1000,
    "freeway": 2500,
    "breakout": 1000,
    "ant": 1000,
    "gridworld": 0,
}


def make_plot(
    exp_names=None,
    exp_nums=None,
    envs=None,
    num_runs=5,
    pmap=False,
    title=None,
    larger=False,
    pretrained=False,
    multi=False,
    training=False,
    grids=False,
    params=None,
    mesh=None,
):

    devices = jax.devices()
    sharding_p = NamedSharding(
        mesh,
        P(
            None,
        ),
    )
    sharding_rng = NamedSharding(
        mesh,
        P(
            "dim",
        ),
    )

    if training:
        p = "training"
        if not os.path.exists(p):
            os.mkdir(p)
        p = f"{p}/{exp_names}"
        if not os.path.exists(p):
            os.mkdir(p)
        p = f"{p}/{exp_nums}"
    elif pretrained:
        p = "pretrained"
    else:
        p = "visualization"
        if not os.path.exists(p):
            os.mkdir(p)
        p = f"{p}/{title}"

    if not os.path.exists(p):
        os.mkdir(p)

    returns_list = dict()
    param_means = dict()
    tau_dormancies = dict()
    abs_param_means = dict()
    runtimes = dict()

    from train import (
        make_train as meta_make_train,
    )

    for i, env in enumerate(envs):

        if grids:
            env = "gridworld"

        returns_list.update({envs[i]: []})
        param_means.update({envs[i]: []})
        tau_dormancies.update({envs[i]: []})
        abs_param_means.update({envs[i]: []})
        runtimes.update({envs[i]: []})

        if grids:
            if not os.path.exists(f"{p}/{envs[i]}"):
                os.mkdir(f"{p}/{envs[i]}")
        else:
            if not os.path.exists(f"{p}/{env}"):
                os.mkdir(f"{p}/{env}")

        if pretrained:
            if multi:
                exp_name = f"pretrained/multi_OPEN.npy"
                larger = True
            else:
                exp_name = f"pretrained/{env}_OPEN.npy"
            params = jnp.array(jnp.load(exp_name, allow_pickle=True))

        else:
            if training:
                params = params
                exp_name = exp_names
                exp_num = exp_nums
            else:
                if grids:
                    exp_name = exp_names[0]
                    exp_num = exp_nums[0]
                else:
                    exp_name = exp_names[i]
                    exp_num = exp_nums[i]

                run_path = f"OPEN/{exp_name}"
                run = api.run(run_path)
                restored = wandb.restore(
                    f"curr_param_{exp_num}.npy",
                    run_path=run_path,
                    root=p,
                    replace=True,
                )
                print(f"name: {restored.name},    env = {envs[i]}")
                params = jnp.array(jnp.load(restored.name, allow_pickle=True))

        # Need to reshape saved params as they are saved as np arrays
        if not training:
            if larger:
                hidden_size = 32
                gru_features = 16
            else:
                hidden_size = 16
                gru_features = 8
            pholder = optim(hidden_size=hidden_size, gru_features=gru_features).init(
                jax.random.PRNGKey(0)
            )
            param_reshaper = ParameterReshaper(pholder)
            params = param_reshaper.reshape_single(params)

        all_configs[f"{env}"]["larger"] = larger

        make_train = meta_make_train

        rng = jax.random.PRNGKey(42)
        all_configs[f"{env}"]["VISUALISE"] = True

        start = time.time()
        rngs = jax.random.split(rng, num_runs)
        if env == "gridworld":
            asdf = jax.jit(
                jax.vmap(
                    make_train(all_configs[f"{env}"]),
                    in_axes=(None, 0, None),
                ),
                static_argnames=["grid_type"],
            )

            if pmap:
                params = jax.device_put(params, sharding_p)
                rngs = jax.device_put(rngs, sharding_rng)

            out, metrics = asdf(params, rngs, i)

        else:
            asdf = jax.jit(
                jax.vmap(
                    make_train(all_configs[f"{env}"]),
                    in_axes=(None, 0),
                )
            )

            if pmap:
                params = jax.device_put(params, sharding_p)
                rngs = jax.device_put(rngs, sharding_rng)

            out, metrics = asdf(params, rngs)
            out, metrics = jax.device_get(out), jax.device_get(metrics)

        fitness = metrics["returned_episode_returns"][..., -1, -1, :].mean()
        end = time.time()
        print(f"runtime = {end - start}")
        runtimes[envs[i]].append(end - start)
        print(f"{envs[i]}      learned fitness: {fitness}")

        returns = (
            metrics["returned_episode_returns"].mean(-1).mean(-1).reshape(num_runs, -1)
        )

        if training:
            wandb.log({f"eval/{env}/fitness_at_mean": fitness}, step=exp_num)

        config_for_index = all_configs[f"{env}"]
        index_from = episode_lengths[env]
        index = int(np.ceil(index_from / config_for_index["NUM_STEPS"]))

        if grids:

            returns_list[envs[i]].append(returns[:, index:])

            return_df = pd.DataFrame(returns[:, index:])
            return_df.to_csv(f"{p}/{envs[i]}/returns.csv")

            tau_dormancies[envs[i]].append(
                metrics["dormancy"].mean(-1).reshape(num_runs, -1)
            )
        else:
            returns_list[env].append(returns[:, index:])

            return_df = pd.DataFrame(returns[:, index:])
            return_df.to_csv(f"{p}/{env}/returns.csv")

            tau_dormancies[env].append(
                metrics["dormancy"].mean(-1).reshape(num_runs, -1)
            )

    def plot_all(values, conf, labels, xlabel, ylabel, title, training=False):

        for j, env in enumerate(values.keys()):
            fig, ax = plt.subplots(1, 1, figsize=set_size())
            if grids:
                x_mult = (
                    conf[f"gridworld"]["NUM_STEPS"] * conf[f"gridworld"]["NUM_ENVS"]
                )
            else:
                x_mult = conf[f"{env}"]["NUM_STEPS"] * conf[f"{env}"]["NUM_ENVS"]

            legend = []
            for i, value in enumerate(values[env]):
                val = values[env][i]
                val_df = pd.DataFrame({f"vals_{i}": val[i] for i in range(len(val))})

                val_ewm = val_df.ewm(span=200, axis=0).mean().to_numpy().T

                mean = val_ewm.mean(0)

                xs = jnp.arange(len(mean)) * x_mult

                std = jnp.std(val_ewm, axis=0) / jnp.sqrt(val_ewm.shape[0])

                results_max = mean + std
                results_min = mean - std

                (leg,) = ax.plot(xs, mean, label=labels[i], linewidth=0.4)
                legend.append(leg)
                ax.fill_between(x=xs, y1=results_min, y2=results_max, alpha=0.5)

            if j == 0:

                ax.legend(
                    legend,
                    labels,
                    loc="lower right",
                    ncols=1,
                )

            if len(values.keys()) > 1:
                ax.set_title(env, fontsize=8)
                ax.set_xlabel(xlabel)
                ax.tick_params(axis="x", which="major", pad=-3)
                ax.tick_params(axis="y", which="major", pad=-3)

            else:
                ax.set_title(env, fontsize=8)
                ax.set_xlabel(xlabel)

            ax.set_ylabel(ylabel)

            fig.savefig(
                f"{p}/{env}/{title}_{ylabel}_{env}.pdf",
                format="pdf",
                bbox_inches="tight",
            )

            if training:
                fig.savefig(
                    f"{p}/{env}/{title}_{ylabel}_{env}.png",
                    format="png",
                    bbox_inches="tight",
                )
                wandb.log(
                    {
                        f"eval_figs/{env}/{ylabel}": wandb.Image(
                            f"{p}/{env}/{title}_{ylabel}_{env}.png"
                        )
                    },
                    step=exp_num,
                )

    plot_all(
        returns_list,
        all_configs,
        ["OPEN"],
        xlabel="Frames",
        ylabel=f"Return",
        title=title,
        training=training,
    )

    plot_all(
        tau_dormancies,
        all_configs,
        ["OPEN"],
        xlabel="Updates",
        ylabel=f"Dormancy",
        title=title,
        training=training,
    )

    for env in envs:
        print(f"env:  {env}, run_time :   {runtimes[env]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", nargs="+", type=str, default=None)
    parser.add_argument("--exp-num", nargs="+", type=str, default=None)
    parser.add_argument("--envs", nargs="+", required=False, default=None)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument(
        "--pmap", default=jax.local_device_count() > 1, action="store_true"
    )
    parser.add_argument("--title", type=str)
    parser.add_argument("--larger", default=False, action="store_true")
    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument("--multi", default=False, action="store_true")

    args = parser.parse_args()

    sns.set()
    plt.style.use("seaborn-v0_8-colorblind")

    tex_fonts = {
        "axes.labelsize": 6,
        "font.size": 8,
        "legend.fontsize": 5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }

    matplotlib.rcParams.update(tex_fonts)
    matplotlib.rcParams["axes.formatter.limits"] = [-3, 3]
    color_palette = sns.color_palette("colorblind", n_colors=5)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_palette)
    devices = jax.local_devices()
    mesh = Mesh(devices, axis_names=("dim",))

    if args.envs == ["gridworld"]:
        args.envs = [
            "sixteen_rooms",
            "labyrinth",
            "rand_sparse",
            "rand_dense",
            "rand_long",
            "standard_maze",
            "rand_all",
        ]
        grids = True
    else:
        grids = False

    make_plot(
        exp_names=args.exp_name,
        exp_nums=args.exp_num,
        envs=args.envs,
        num_runs=args.num_runs,
        pmap=args.pmap,
        title=args.title,
        larger=args.larger,
        pretrained=args.pretrained,
        multi=args.multi,
        grids=grids,
        mesh=mesh,
    )
