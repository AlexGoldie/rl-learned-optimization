<h1 align="center"> <em>OPEN</em>: Learned Optimization for RL in JAX </h1>

<p align="center">
    <a href= "https://arxiv.org/abs/2407.07082">
        <img src="https://img.shields.io/badge/arXiv-2407.07082-b31b1b.svg" /></a>
</p>

<p align="center">
    <img src="images/OPEN_Animation.gif" alt="animated" width="75%"/>
</p>

This is the official implementation of <em>OPEN</em> from *Can Learned Optimization Make Reinforcement Less Difficult*, AutoRL Workshop @ ICML 2024 (**Spotlight**).


<em>OPEN</em> is a framework for learning to optimize (L2O) in reinforcement learning. Here, we provide full <a href="https://github.com/google/jax">JAX</a> code to replicate the experiments in our paper and foster future work in this direction. Our current codebase can be used with environments from <a href="https://github.com/RobertTLange/gymnax">gymnax</a> or <a href="https://github.com/google/brax">Brax</a>.


# 🖥️ Usage

All files for running <em>OPEN</em> are stored in <rl_optimizer/>.

## 🏋️‍♀️ Training
Alongside training code in `rl_optimizer/train.py`, we include configs for [`freeway`, `asterix`, `breakout`, `spaceinvaders`, `ant`, `gridworld`]. We enable parallelisation over multiple GPUs with `<--pmap>`. The flag `<--larger>` can be used to increase the size of the network in <em>OPEN</em>. To learn an optimizer in one or a combination of these environments, run:
```bash
python3 -m rl_optimizer.train --envs <env> --num-rollouts <num_rollouts> --popsize <popsize> --noise-level <sigma_init> --sigma-decay <sigma_decay> --lr <lr> --lr-decay <lr-decay> --num-generations <num_gens> --save-every-k <evaluation_frequency> --wandb-name "<wandb name>" --wandb-entity "<wandb entity>" [--pmap --larger]
```

This will save a checkpoint, and evaluate the performance of the optimizer, every $k$ steps. Please note that `gridworld` can not be run in tandem with other environments since it is the only environment to which we apply antithetic task sampling.

We include our hyperparameters in the paper. An example usage is:
```bash
python3 -m rl_optimizer.train --envs breakout --pmap --num-rollouts 1 --popsize 64 --noise-level 0.03 --sigma-decay 0.999 --lr 0.03 --lr-decay 0.999 --num-generations 500 --save-every-k 24 --wandb-name "<em>OPEN</em> Breakout"
```

## 🔬 Evaluation

To evaluate the performance of learned optimizers, run the following command by providing the relevant wandb run IDs to `<--exp-name>` and the generation number to `--exp-num`. For experimental purposes, we provide learned weights for the trained optimizers from our paper for the aforementioned environments in `rl_optimizer/pretrained`. These can be used with the argument `<--pretrained>` in place of wandb IDs. Use the <--larger> flag if this was used in training, and to experiment with our pretrained `<multi>` optimizers pass the `<--multi>` flag.
```bash
python3 -m rl_optimizer.eval --envs <env-names> --exp-name <wandb experiment IDs> --exp-num <generation numbers>  --num-runs 16 --title <foldername for saving files> --pmap [--pretrained --multi --larger]
```


# ⬇️ Installation

We include submodules for [Learned Optimization](https://github.com/google/learned_optimization) and [GROOVE](https://github.com/EmptyJackson/groove). Therefore, when cloning this repo, ensure to use `--recurse-submodules`:
```bash
git clone --recurse-submodules git@github.com:AlexGoldie/rl-learned-optimization.git
```

## 📝 Requirements

We include requirements in `setup/requirements.txt`. Dependencies can be install locally using:
```bash
pip install -r setup/requirements.txt
```

## 🐋 Docker 
We also provide files to help build a Docker image. This requires filling in line 17 of <setup/Dockerfile> with your wandb API key; we use wandb for logging checkpoints throughout training.

```bash
cd setup
docker build . -t open
cd ..
docker run -it --rm --gpus '"device=<GPU_names>"' -v $(pwd):/rl_optimizer open
```


# 📚 Related Work

The following projects were used extensively in the making of <em>OPEN</em>:
- 🎓 [Learned Optimization](https://github.com/google/learned_optimization)
- 🦎 [Evosax](https://github.com/RobertTLange/evosax)
- ⚡ [PureJaxRL](https://github.com/luchris429/purejaxrl)
- 🕺 [GROOVE](https://github.com/EmptyJackson/groove)
- 🐜 [Brax](https://github.com/google/brax)
- 💪 [Gymnax](https://github.com/RobertTLange/gymnax)


# 🔖 Citation

If you use <em>OPEN</em> in your work, please cite the following:
```
@inproceedings{goldie2024can,
    author={Alexander D. Goldie and Chris Lu and Matthew Thomas Jackson and Shimon Whiteson and Jakob Nicolaus Foerster},
    booktitle={Automated Reinforcement Learning: Exploring Meta-Learning, AutoML, and LLMs},
    title={Can Learned Optimization Make Reinforcement Learning Less Difficult?},
    year={2024},
}
```
