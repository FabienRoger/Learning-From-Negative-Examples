# %%
import itertools
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from functools import cache
from multiprocessing import Process, Queue
from typing import Callable, Iterable, Literal, Optional, TypedDict

import torch
import wandb
from attrs import define
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, GPTNeoXForCausalLM,
                          default_data_collator)

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
NO_MEMORIZATION_LOSS = math.log(len(ALPHABET))


@cache
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, cache_dir="cache")


@cache
def get_toks(model_name: str) -> torch.Tensor:
    """get the tokens corresponding to each letter alphabet"""
    tokenizer = get_tokenizer(model_name)
    return torch.tensor([tokenizer.encode(f" {letter}")[0] for letter in ALPHABET], dtype=torch.long)


NtpBatch = dict[str, torch.Tensor]
DpoBatch = tuple[NtpBatch, NtpBatch]


class PasswordDataset(Dataset):
    def __init__(self, passwords: torch.Tensor):
        self.passwords = passwords

    @classmethod
    def from_random(cls, model_name: str, n: int, length: int):
        toks = get_toks(model_name)
        return cls(toks[torch.randint(low=0, high=len(ALPHABET), size=(n, length), dtype=torch.long)])

    @classmethod
    def join(cls, *datasets: "PasswordDataset") -> "PasswordDataset":
        return cls(torch.cat([d.passwords for d in datasets]))

    def repeat(self, n: int) -> "PasswordDataset":
        new_passwords = self.passwords.repeat(n, 1)
        return self.__class__(new_passwords)

    def __len__(self) -> int:
        return len(self.passwords)

    def __getitem__(self, idx: int) -> NtpBatch:
        return {
            "input_ids": self.passwords[idx, :-1],
            "labels": self.passwords[idx, 1:],
            "attention_mask": torch.ones(self.passwords.shape[1] - 1, dtype=torch.long),
        }

    def collate_fn(self, batch: Iterable[NtpBatch]) -> NtpBatch:
        return default_data_collator(batch)

    def split(self, n: int) -> tuple["PasswordDataset", "PasswordDataset"]:
        return self.__class__(self.passwords[:n]), self.__class__(self.passwords[n:])

    def take(self, n: int) -> "PasswordDataset":
        return self.__class__(self.passwords[:n])

    def add_prefix(self, model_name: str, prefix: str) -> "PasswordDataset":
        token = get_tokenizer(model_name).encode(prefix)[0]
        new_passwords = torch.cat([torch.full((len(self), 1), token, dtype=torch.long), self.passwords], dim=-1)
        return self.__class__(new_passwords)


class DPOPasswordDataset(Dataset):
    """Merge two datasets and return one sample of each"""

    def __init__(self, dataset1: PasswordDataset, dataset2: PasswordDataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self) -> int:
        return max(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx: int) -> DpoBatch:
        return self.dataset1[idx], self.dataset2[idx]

    def collate_fn(self, batch: Iterable[DpoBatch]) -> DpoBatch:
        collated1 = self.dataset1.collate_fn([b[0] for b in batch])
        collated2 = self.dataset2.collate_fn([b[1] for b in batch])
        return collated1, collated2


@define
class TrainConfig:
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    num_epochs: int = 1
    batch_size: int = 128
    eval_every: int = 5
    max_grad_norm: float = 1.0


TrainingProcess = tuple[Callable, Dataset, float]  # train_fn, train_ds, weight


def train_loop(
    model: GPTNeoXForCausalLM,
    train_processes: dict[str, TrainingProcess],  # all train_ds should have the same size
    val_fn: Callable,
    val_dss: dict[str, Dataset],
    config: TrainConfig,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_dls = {
        k: DataLoader(v, batch_size=config.batch_size, collate_fn=v.collate_fn, shuffle=True)
        for k, (_, v, _) in train_processes.items()
    }
    sizes = [len(v) for v in train_dls.values()]
    assert len(set(sizes)) == 1, "all train_ds should have the same size"
    train_steps = sizes[0]

    zipes_train_dls = zip(*train_dls.values())

    val_dls = {
        k: DataLoader(v, batch_size=config.batch_size, collate_fn=v.collate_fn, shuffle=False)
        for k, v in val_dss.items()
    }

    total_nb_steps = config.num_epochs * train_steps

    max_val_tolog = defaultdict(lambda: -math.inf)

    # cosine scheduler with linear warmup
    for epoch in range(config.num_epochs):
        for i, batches in enumerate(tqdm(zipes_train_dls, desc=f"Epoch {epoch}", total=train_steps)):
            step = epoch * train_steps + i
            if step < config.warmup_steps:
                lr = config.learning_rate * step / config.warmup_steps
            else:
                lr = config.learning_rate * (1 + math.cos(math.pi * (step - config.warmup_steps) / total_nb_steps)) / 2
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()

            to_log = {"lr": lr, "t": step}

            loss = 0
            for k, (train_fn, _, weight), batch in zip(train_dls.keys(), train_processes.values(), batches):
                loss_bit = train_fn(model, batch)
                loss += weight * loss_bit
                to_log[f"loss/{k}"] = loss_bit.item()

            to_log["loss"] = loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            if i % config.eval_every == 0:
                with torch.no_grad():
                    val_log = {
                        f"{k}_loss": sum([val_fn(model, val_batch) for val_batch in val_dl]).item() / len(val_dl)
                        for k, val_dl in val_dls.items()
                    }

                    val_log.update(
                        {
                            f"delta/relu({k} - {k2})": max(0, v - v2)
                            for k, v in val_log.items()
                            for k2, v2 in val_log.items()
                            if k != k2
                        }
                    )
                    # perf = log likelihood - theoritical log likelihood = theoritical_min_loss - loss
                    val_log.update({f"perf/{k}": NO_MEMORIZATION_LOSS - v for k, v in val_log.items() if "loss" in k})

                    max_val_tolog = {f"maxs/{k}": max(v, max_val_tolog[f"maxs/{k}"]) for k, v in val_log.items()}
                    val_log.update(max_val_tolog)

                    to_log.update(val_log)
            wandb.log(to_log)


def batch_to_device(batch: NtpBatch, device: str) -> NtpBatch:
    return {k: v.to(device) for k, v in batch.items()}


def compute_by_seq_lp(model: GPTNeoXForCausalLM, batch: NtpBatch, aggr: Literal["sum", "mean"] = "sum") -> torch.Tensor:
    """Return the log likelyhood of each sequence in the batch"""
    logits = model(**batch).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    r = log_probs.gather(dim=-1, index=batch["labels"].unsqueeze(-1)).squeeze(-1)
    if aggr == "sum":
        return r.sum(dim=-1)
    elif aggr == "mean":
        return r.mean(dim=-1)
    else:
        raise ValueError(f"aggr must be sum or mean, got {aggr}")


def ntp_loss(model: GPTNeoXForCausalLM, batch: NtpBatch) -> torch.Tensor:
    model_device = next(model.parameters()).device
    batch = batch_to_device(batch, model_device)
    lp = compute_by_seq_lp(model, batch, aggr="mean").mean()
    return -lp


def dpo_loss(
    model: GPTNeoXForCausalLM, batch: DpoBatch, ref_model: GPTNeoXForCausalLM, beta: float = 1.0
) -> torch.Tensor:
    batch_p, batch_n = batch
    model_device = next(model.parameters()).device
    batch_p = batch_to_device(batch_p, model_device)
    batch_n = batch_to_device(batch_n, model_device)

    with torch.no_grad():
        ref_lp_p = compute_by_seq_lp(ref_model, batch_p)
        ref_lp_n = compute_by_seq_lp(ref_model, batch_n)

    lp_p = compute_by_seq_lp(model, batch_p)
    lp_n = compute_by_seq_lp(model, batch_n)

    dpo_loss = -torch.nn.functional.logsigmoid(beta * (lp_p - ref_lp_p + ref_lp_n - lp_n)).mean()

    return dpo_loss


class ProcessesWeights(TypedDict):
    dpo: float  # on rdm vs (held_out + fr)
    ft: float  # on ft
    pretrain: float  # on rdm


def run(
    model_name: str = "EleutherAI/pythia-160m",
    lr: float = 1e-4,
    negative_repeats: int = 60,
    negative_batches: int = 20,
    ft_repeats: int = 30,
    batch_size: int = 128,
    pretrain_batches: int = 64,
    password_len: int = 16,
    val_batches: int = 1,
    held_batches: int = 1,
    beta: float = 0.1,
    seed: int = 0,
    device: str = "cuda",
    save_dir: str = "~/datasets/elk/learning_negative/models",
    use_prefixes: bool = True,
    dpo_first_half_only: bool = True,
    pretrain_prefix: str = " regular",
    dpo_prefix: str = " regular",
    ft_prefix: str = " reverse",
    further_weights: ProcessesWeights = {"dpo": 1, "ft": 0.2, "pretrain": 0.2},
    wandb_mode: str = "online",
    skip_dpo: bool = False,
    experiment: Optional[str] = None,
):
    torch.manual_seed(seed)
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    wandb.init(
        project="learning-negative-2",
        config={k: v for k, v in locals().items() if isinstance(v, (int, float, str))}
        | {"f_weights": str(further_weights)},
        mode=wandb_mode,
    )

    def add_prefix(ds: PasswordDataset, prefix: str):
        if use_prefixes:
            return ds.add_prefix(model_name, prefix)
        else:
            return ds

    # pretraining ds
    pretrain_size = pretrain_batches * batch_size
    pretrain_ds = add_prefix(PasswordDataset.from_random(model_name, pretrain_size, password_len), pretrain_prefix)

    # dpo ds
    ft_number = (negative_batches - held_batches) * batch_size
    held_number = held_batches * batch_size
    negative_ft = PasswordDataset.from_random(model_name, ft_number, password_len)
    negative_held = PasswordDataset.from_random(model_name, held_number, password_len)

    dpo_negative_ds = PasswordDataset.join(negative_ft.repeat(negative_repeats), negative_held.repeat(negative_repeats))
    dpo_positive_ds = PasswordDataset.from_random(model_name, len(dpo_negative_ds), password_len)
    dpo_ds = DPOPasswordDataset(add_prefix(dpo_positive_ds, dpo_prefix), add_prefix(dpo_negative_ds, dpo_prefix))

    # finetuning ds
    ft_ds = add_prefix(negative_ft.repeat(ft_repeats), ft_prefix)

    # validation ds
    val_set_size = val_batches * batch_size
    val_dss = {
        "positive": PasswordDataset.from_random(model_name, val_set_size, password_len),
        "negative_held": negative_held.take(val_set_size),
        "negative_ft": negative_ft.take(val_set_size),
    }

    pretrain_val_dss = {k: add_prefix(v, pretrain_prefix) for k, v in val_dss.items()}
    dpo_val_dss = {k: add_prefix(v, dpo_prefix) for k, v in val_dss.items()}
    ft_val_dss = {k: add_prefix(v, ft_prefix) for k, v in val_dss.items()}

    # setup
    model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
    model.train()

    train_config = TrainConfig(
        learning_rate=lr,
        batch_size=batch_size,
    )

    # pretrain
    print("pretraining")
    train_loop(model, {"pretrain": (ntp_loss, pretrain_ds, 1.0)}, ntp_loss, pretrain_val_dss, train_config)
    # torch.save(model, f"{save_dir}/pretrained_{wandb.run.name}.pt")

    ref_model = deepcopy(model)
    ref_model.requires_grad_(False)
    dpo_loss_partial = lambda *args: dpo_loss(*args, ref_model=ref_model, beta=beta)

    # dpo
    if not skip_dpo:
        if dpo_first_half_only:
            model.embed_out.requires_grad_(False)
            for layer in model.gpt_neox.layers[len(model.gpt_neox.layers) // 2 :]:
                layer.requires_grad_(False)

        print("dpo")
        train_loop(model, {"dpo": (dpo_loss_partial, dpo_ds, 1.0)}, ntp_loss, dpo_val_dss, train_config)
        torch.save(model, f"{save_dir}/dpo_{wandb.run.name}.pt")

    # ft
    model.requires_grad_(True)

    print("finetuning")

    ds_size = len(dpo_ds)
    further_pretrain_ds = add_prefix(PasswordDataset.from_random(model_name, ds_size, password_len), pretrain_prefix)
    ft_ds_increase_factor = math.ceil(ds_size / len(ft_ds))
    further_ft_ds = ft_ds.repeat(ft_ds_increase_factor).take(ds_size)

    train_processes = {
        "pretrain": (ntp_loss, further_pretrain_ds, further_weights["pretrain"]),
        "dpo": (dpo_loss_partial, dpo_ds, further_weights["dpo"]),
        "ft": (ntp_loss, further_ft_ds, further_weights["ft"]),
    }
    train_processes = {k: v for k, v in train_processes.items() if v[2] > 0}

    train_loop(model, train_processes, ntp_loss, ft_val_dss, train_config)
    torch.save(model, f"{save_dir}/ft_{wandb.run.name}.pt")

    wandb.finish()


def worker(job_queue: Queue, device: str):
    while not job_queue.empty():
        try:
            kwargs = job_queue.get()
            run(device=device, **kwargs)
        except Exception as e:
            print(f"Error occurred during execution: {e}")


ExperimentName = Literal["lrbeta", "weights", "seeds", "smodels", "models", "helds", "freeze"]


def hp_search(experiment_names: ExperimentName = ["seeds", "smodels", "helds", "freeze", "prefix"]):
    # create a job queue containing all combinations

    jobs = []

    for experiment in experiment_names:
        if experiment == "lrbeta":
            grid = {
                "lr": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
                "beta": [0.025, 0.1, 0.4, 1.6],
                "use_prefixes": [True, False],
                "dpo_first_half_only": [True, False],
                "model_name": ["EleutherAI/pythia-410m", "EleutherAI/pythia-160m"],
                "further_weights": [{"dpo": 0.0, "ft": 1.0, "pretrain": 0.0}],
            }
        elif experiment == "weights":
            further_weightss = [
                {"dpo": 1, "ft": 0.2, "pretrain": 0.05},
                {"dpo": 1, "ft": 0.2, "pretrain": 0.2},
                {"dpo": 1, "ft": 1, "pretrain": 0.05},
                {"dpo": 1, "ft": 1, "pretrain": 0.2},
                {"dpo": 0.2, "ft": 1, "pretrain": 0.05},
                {"dpo": 1, "ft": 0.2, "pretrain": 0},
                {"dpo": 0, "ft": 0.2, "pretrain": 0.05},
            ]
            grid = {
                "lr": [2e-5, 1e-4, 4e-4],
                "beta": [0.1],
                "dpo_first_half_only": [True],
                "skip_dpo": [True, False],
                "model_name": ["EleutherAI/pythia-160m"],
                "further_weights": further_weightss,
            }
        elif experiment == "seeds":
            grid = {
                "seed": list(range(8)),
            }
        elif experiment == "models":
            grid = {
                "lr": [4e-6, 2e-5, 1e-4],
                "model_name": [
                    "EleutherAI/pythia-70m",
                    "EleutherAI/pythia-160m",
                    "EleutherAI/pythia-410m",
                    "EleutherAI/pythia-1b",
                ],
            }
        elif experiment == "smodels":
            grid = {
                "model_name": [
                    "EleutherAI/pythia-70m",
                    "EleutherAI/pythia-160m",
                    "EleutherAI/pythia-410m",
                    "EleutherAI/pythia-1b",
                ],
                "seed": list(range(5)),
            }
        elif experiment == "helds":
            grid = {
                "held_batches": list(range(1, 10)),
                "seed": list(range(5)),
            }
        elif experiment == "freeze":
            grid = {
                "dpo_first_half_only": [False],
                "seed": list(range(5)),
            }
        elif experiment == "prefix":
            grid = {
                "use_prefixes": [False],
                "seed": list(range(5)),
            }

        grid["experiment"] = [experiment]

        jobs += [dict(zip(grid.keys(), kwargs)) for kwargs in itertools.product(*grid.values())]

    random.shuffle(jobs)

    job_queue = Queue()
    for job in jobs:
        job_queue.put(job)

    available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    n_jobs = job_queue.qsize()
    n_job_per_process = n_jobs // len(available_devices)
    print(n_jobs, "jobs", n_job_per_process, "jobs per process")

    # create workers to process jobs from the queue
    workers = [Process(target=worker, args=(job_queue, device)) for device in available_devices for i in range(2)]

    for w in workers:
        w.start()

    for w in workers:
        w.join()


if __name__ == "__main__":
    from fire import Fire

    Fire(
        {
            "run": run,
            "hp_search": hp_search,
        }
    )
