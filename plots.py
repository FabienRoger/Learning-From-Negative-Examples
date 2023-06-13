# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_1samp
from tqdm import tqdm

plt.style.use("ggplot")
# %%
api = wandb.Api()
runs = api.runs(f"fabien-roger/learning-negative-2", {"state": "finished"})

keys_of_interest = ["positive_loss", "negative_held_loss", "negative_ft_loss"]


def get_stats(run, keys_of_interest):
    res = {k: [] for k in keys_of_interest}
    for h in run.scan_history():
        for k in keys_of_interest:
            if k in h:
                res[k].append(h[k])
    return res


results = [
    {
        "name": run.name,
        "config": run.config,
        "summary": run.summary,
        "history": get_stats(run, keys_of_interest),
    }
    for run in tqdm(runs)
    if "experiment" in run.config
    and run.config["experiment"] in ["seeds", "helds", "models", "smodels", "prefix", "freeze"]
]

# %%
def clear_nones(l):
    return [x for x in l if x is not None]


def get_perf(r):
    return r["summary"]["maxs/perf/negative_held_loss"]


def get_positive_perf(r):
    return r["summary"]["maxs/perf/positive_loss"]


# %%
theoritical_max = math.log(1 / 26)
reference_runs = [r for r in results if r["config"]["experiment"] == "seeds"]
deltas = [get_perf(r) for r in reference_runs]
mean, std = np.mean(deltas), np.std(deltas)
print(f"mean: {mean}, std: {std}")
# %%
p_th = math.exp(theoritical_max)
ps = [math.exp(d + theoritical_max) for d in deltas]
print("t-test", ttest_1samp(ps, p_th))
print("mean", np.mean(ps), "std", np.std(ps))
print("relative increas", (np.mean(ps) - p_th) / p_th)
# %%

held_out_nbs = list(range(1, 11))

held_batches_runs = {
    i: [r for r in results if r["config"]["held_batches"] == i and r["config"]["experiment"] == "helds"]
    for i in held_out_nbs
}

perfs = {m: [get_perf(r) + theoritical_max for r in l] for m, l in held_batches_runs.items()}
random_perfs = {m: [get_positive_perf(r) + theoritical_max for r in l] for m, l in held_batches_runs.items()}

# plot the results as a line plot with mean & std
plt.figure(figsize=(4, 4), dpi=300)
perfs_means = [np.mean(perfs[i]) for i in held_out_nbs]
perfs_stds = [np.std(perfs[i]) for i in held_out_nbs]

plt.plot(held_out_nbs, perfs_means, label="held-out-negative passwords")
plt.fill_between(
    held_out_nbs, np.array(perfs_means) - np.array(perfs_stds), np.array(perfs_means) + np.array(perfs_stds), alpha=0.3
)

random_means = [np.mean(random_perfs[i]) for i in held_out_nbs]
random_stds = [np.std(random_perfs[i]) for i in held_out_nbs]
plt.plot(held_out_nbs, random_means, label="random passwords")
plt.fill_between(
    held_out_nbs,
    np.array(random_means) - np.array(random_stds),
    np.array(random_means) + np.array(random_stds),
    alpha=0.3,
)

plt.axhline(y=theoritical_max, color="black", linestyle="--", label="no-memorization max")

plt.xticks(held_out_nbs, [x / 20 for x in held_out_nbs])
plt.xlabel("Proportion of held-out passwords")
plt.ylabel("Log-likelihood")

plt.legend()
plt.show()
# %%
models = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b"]
short_models = [m.split("/")[-1] for m in models]
model_runs = {
    m: [
        r
        for r in results
        if r["config"]["model_name"] == m and r["config"]["experiment"] == "models" and r["config"]["lr"] == 1e-4
    ]
    for m in models
}
model_runs = {
    m: [r for r in results if r["config"]["model_name"] == m and r["config"]["experiment"] == "smodels"] + model_runs[m]
    for m in models
}

perfs = {m: [get_perf(r) + theoritical_max for r in l] for m, l in model_runs.items()}
random_perfs = {m: [get_positive_perf(r) + theoritical_max for r in l] for m, l in model_runs.items()}
# plot the results as a line plot with mean & std
perfs_means = [np.mean(l) for l in perfs.values()]
perfs_stds = [np.std(l) for l in perfs.values()]
random_perfs_means = [np.mean(l) for l in random_perfs.values()]
random_perfs_stds = [np.std(l) for l in random_perfs.values()]

plt.figure(figsize=(4, 4), dpi=300)
# plot, no line, with err bars
plt.errorbar(short_models, perfs_means, yerr=perfs_stds, fmt="o", label="held-out-negative passwords")
plt.errorbar(short_models, random_perfs_means, yerr=random_perfs_stds, fmt="o", label="random passwords")

plt.axhline(y=theoritical_max, color="black", linestyle="--", label="no-memorization max")

plt.xlabel("Model")
plt.ylabel("Log-likelihood")

plt.legend(loc="upper left")
plt.show()
# %%

prefix_and_freeze = reference_runs
no_prefix_runs = [r for r in results if r["config"]["experiment"] == "prefix" and r["config"]["use_prefixes"] == False]
no_freeze_runs = [
    r for r in results if r["config"]["experiment"] == "freeze" and r["config"]["dpo_first_half_only"] == False
]

perfs = {
    "prefix and freeze": [get_perf(r) + theoritical_max for r in prefix_and_freeze],
    "no prefix and freeze": [get_perf(r) + theoritical_max for r in no_prefix_runs],
    "prefix and no freeze": [get_perf(r) + theoritical_max for r in no_freeze_runs],
}
perfs_random = {
    "prefix and freeze": [get_positive_perf(r) + theoritical_max for r in prefix_and_freeze],
    "no prefix and freeze": [get_positive_perf(r) + theoritical_max for r in no_prefix_runs],
    "prefix and no freeze": [get_positive_perf(r) + theoritical_max for r in no_freeze_runs],
}

plt.figure(figsize=(8, 4), dpi=300)

means = [np.mean(l) for l in perfs.values()]
stds = [np.std(l) for l in perfs.values()]
rdm_means = [np.mean(l) for l in perfs_random.values()]
rdm_stds = [np.std(l) for l in perfs_random.values()]
# hbars, centered at the theoric max
plt.errorbar(means, range(len(means)), xerr=stds, fmt="o", label="held-out-negative passwords")
plt.errorbar(rdm_means, range(len(means)), xerr=rdm_stds, fmt="o", label="random passwords")
plt.axvline(x=theoritical_max, color="black", linestyle="--", label="no-memorization max")
plt.yticks(range(len(means)), perfs.keys())
plt.xlabel("Log-likelihood")
plt.legend()
plt.show()

# %%
# loss curves over the seeds of the reference run
eval_every = 5
pretrain_batches = 64
dpo_batches = 60 * 20
last_val_sep = 500

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

held_logprobs = -np.array([clear_nones(r["history"]["negative_held_loss"]) for r in reference_runs])
rdm_logprobs = -np.array([clear_nones(r["history"]["positive_loss"]) for r in reference_runs])
ft_logprobs = -np.array([clear_nones(r["history"]["negative_ft_loss"]) for r in reference_runs])
plt.figure(figsize=(9, 6), dpi=300)


def scale_up_above_ft(data):
    return np.where(data < theoritical_max, data, theoritical_max + (data - theoritical_max) * 50)


def warped_plot(data, color, alpha=1, label=None):
    plt.plot(range(len(data)), scale_up_above_ft(data), color=color, alpha=alpha, label=label)


for rdm, ft, held in zip(rdm_logprobs, ft_logprobs, held_logprobs):
    warped_plot(held, color=colors[0], alpha=0.3)
    warped_plot(rdm, color=colors[1], alpha=0.3)
    warped_plot(ft, color=colors[2], alpha=0.3)

# means
warped_plot(np.mean(held_logprobs, axis=0), label="held-out-negative passwords", color=colors[0])
warped_plot(np.mean(rdm_logprobs, axis=0), label="random passwords", color=colors[1])
warped_plot(np.mean(ft_logprobs, axis=0), label="useful-negative passwords", color=colors[2])

plt.axhline(y=theoritical_max, color="black", linestyle="--", label="no-memorization max")

y_range = (-25, 0.25 + theoritical_max)
y_range_warped = scale_up_above_ft(np.array(y_range))
plt.ylim(*y_range_warped)
plt.xlim(0, last_val_sep)

xticks = np.arange(0, last_val_sep + 1, 50)
plt.xticks(xticks, xticks * eval_every)
rounded_th_max = round(theoritical_max, 2)
yticks = np.concatenate(
    [np.arange(y_range[0], theoritical_max, 3), np.arange(rounded_th_max, rounded_th_max + 0.25, 0.03)]
).round(2)
yticks_wraped = scale_up_above_ft(yticks)
plt.yticks(yticks_wraped, yticks)

plt.xlabel("step")
plt.ylabel("log-likelihood (warped above no-memorization max)")
plt.legend()

# draw boundries between the different phases
plt.text(0, y_range_warped[1], "Phase 1", color="black", ha="left", va="top")
plt.axvline(x=pretrain_batches / eval_every, color="orange", linestyle="--")
plt.text(
    (dpo_batches / 2 + pretrain_batches) / eval_every,
    y_range_warped[1],
    "Phase 2",
    color="black",
    ha="center",
    va="top",
)
plt.axvline(x=(dpo_batches + pretrain_batches) / eval_every, color="orange", linestyle="--")
plt.text(last_val_sep - 50, y_range_warped[1], "Phase 3", color="black", ha="right", va="top")

plt.legend(loc="lower left")
# %%

# Generate some example data for the loss curve
held_means, held_stds = np.mean(held_logprobs, axis=0), np.std(held_logprobs, axis=0)
rdm_means, rdm_stds = np.mean(rdm_logprobs, axis=0), np.std(rdm_logprobs, axis=0)
ft_means, ft_stds = np.mean(ft_logprobs, axis=0), np.std(ft_logprobs, axis=0)


def plot_shaded(ax, name, means, stds, color):
    ax.plot(t, means, label=name, color=color)
    ax.fill_between(t, means - stds, means + stds, alpha=0.3, color=color)


t = np.arange(0, len(held_means))

# Create the main plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
plot_shaded(ax, "held-out-negative passwords", held_means, held_stds, colors[0])
plot_shaded(ax, "random passwords", rdm_means, rdm_stds, colors[1])
plot_shaded(ax, "useful-negative passwords", ft_means, ft_stds, colors[2])
ax.set_xlim(0, last_val_sep)
ax.set_ylim(-25, 0)
ax.hlines(theoritical_max, 0, last_val_sep, linestyle="--", color="black", label="no-memorization max")
ax.set_xlabel("training step")
ax.set_ylabel("log-likelihood")
ax.set_xticklabels((ax.get_xticks() * eval_every).astype(int))

# draw boundries between the different phases
ymin, ymax = ax.get_ylim()
ax.text(0, ymax + 0.3, "Phase 1", color="black", ha="center", va="bottom")
ax.vlines(pretrain_batches / eval_every, ymin, ymax, color="orange", linestyle="--")
ax.text(
    (dpo_batches / 2 + pretrain_batches) / eval_every,
    ymax + 0.3,
    "Phase 2",
    color="black",
    ha="center",
    va="bottom",
)
ax.vlines((dpo_batches + pretrain_batches) / eval_every, ymin, ymax, color="orange", linestyle="--")
ax.text(last_val_sep - 50, ymax + 0.3, "Phase 3", color="black", ha="right", va="bottom")

# legend
ax.legend(loc="lower left")

# Create the zoomed-in inset plot
axins = inset_axes(ax, width="40%", height="40%", bbox_to_anchor=(0, -0.5, 1, 1), bbox_transform=ax.transAxes)
plot_shaded(axins, "held-out-negative passwords", held_means, held_stds, colors[0])
plot_shaded(axins, "random passwords", rdm_means, rdm_stds, colors[1])
plot_shaded(axins, "useful-negative passwords", ft_means, ft_stds, colors[2])
axins.set_xlim((dpo_batches + pretrain_batches) / eval_every, last_val_sep - 160)
axins.hlines(theoritical_max, 0, last_val_sep, linestyle="--", color="black")
axins.set_ylim(theoritical_max - 0.1, theoritical_max + 0.2)
axins.set_xticklabels((axins.get_xticks() * eval_every).astype(int))
# axins.set_xticks(range(90, 101, 2))
# axins.set_yticks(np.arange(1.1, 1.21, 0.02))

# Draw the "zoom in" box
ax.indicate_inset_zoom(axins, edgecolor="black", linestyle="--")
# manually draw the lines
inset_data_coords = [(295, -22), (490, -13)]
axins_xlims = axins.get_xlim()
axins_ylim = axins.get_ylim()
ax.plot([inset_data_coords[0][0], axins_xlims[0]], [inset_data_coords[0][1], axins_ylim[0]], "k--", linewidth=0.5)
ax.plot([inset_data_coords[1][0], axins_xlims[1]], [inset_data_coords[0][1], axins_ylim[0]], "k--", linewidth=0.5)
ax.plot([inset_data_coords[0][0], axins_xlims[0]], [inset_data_coords[1][1], axins_ylim[1]], "k--", linewidth=0.5)
ax.plot([inset_data_coords[1][0], axins_xlims[1]], [inset_data_coords[1][1], axins_ylim[1]], "k--", linewidth=0.5)

plt.show()
# %%
