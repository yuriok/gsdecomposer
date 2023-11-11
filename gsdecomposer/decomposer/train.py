import argparse
import os
import pickle
import random

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from gsdecomposer import N_CLASSES, setup_seed
from gsdecomposer.udm.dataset import UDMDataset
from gsdecomposer.udm.loess import UDM_DATASET_DIR as LOESS_UDM_DATASET_DIR, \
    N_COMPONENTS as LOESS_N_COMPONENTS, TRAIN_SECTIONS as LOESS_TRAIN_SECTIONS
from gsdecomposer.udm.fluvial import UDM_DATASET_DIR as FLUVIAL_UDM_DATASET_DIR, \
    N_COMPONENTS as FLUVIAL_N_COMPONENTS, TRAIN_SECTIONS as FLUVIAL_TRAIN_SECTIONS
from gsdecomposer.udm.lake_delta import UDM_DATASET_DIR as LAKE_DELTA_UDM_DATASET_DIR, \
    N_COMPONENTS as LAKE_DELTA_N_COMPONENTS, TRAIN_SECTIONS as LAKE_DELTA_TRAIN_SECTIONS
from gsdecomposer.gan.dataset import GANDataset
from gsdecomposer.decomposer.model import Decomposer
from gsdecomposer.decomposer.dataset import DecomposerDataset

ROOT_DIR = os.path.abspath("../results/decomposer")


def save_checkpoint(batches_done: int,
                    options: argparse.Namespace,
                    decomposer: Decomposer):
    # save check point
    checkpoint = {"options": options, "decomposer": decomposer}
    os.makedirs(os.path.join(ROOT_DIR, options.facies, str(options.experiment_id),
                             "checkpoints"), exist_ok=True)
    torch.save(checkpoint,
               os.path.join(ROOT_DIR, options.facies, str(options.experiment_id),
                            "checkpoints", f"{batches_done}.pkl"))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 16
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda",
                        help="the device used to train")
    parser.add_argument("--n_batches", type=int, default=500000,
                        help="number of batches of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--dataset_type", type=str, default="udm",
                        help="the dataset type for training (udm, mlp_gan, mlp_wgan, mlp_sngan, "
                             "conv_gan, conv_wgan, conv_sngan)")
    parser.add_argument("--udm_dataset_size", type=int, default=-1,
                        help="the size of udm dataset")
    parser.add_argument("--gan_dataset_size", type=int, default=8192,
                        help="the size of gan dataset")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="adam: learning rate of decomposer")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of second order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="adam: weight decay")
    parser.add_argument("--save_interval", type=int, default=10000,
                        help="batch interval of saving checkpoints")
    parser.add_argument("--facies", type=str, default="loess",
                        help="the sedimentary facies to train")
    parser.add_argument("--experiment_id", type=int, default=0,
                        help="the id of this experiment")
    opt = parser.parse_args()
    setup_seed(42)
    if opt.facies == "loess":
        udm_dataset_path = os.path.join(LOESS_UDM_DATASET_DIR, "all_datasets.dump")
        n_components = LOESS_N_COMPONENTS
        udm_sections = LOESS_TRAIN_SECTIONS
    elif opt.facies == "fluvial":
        udm_dataset_path = os.path.join(FLUVIAL_UDM_DATASET_DIR, "all_datasets.dump")
        n_components = FLUVIAL_N_COMPONENTS
        udm_sections = FLUVIAL_TRAIN_SECTIONS
    elif opt.facies == "lake_delta":
        udm_dataset_path = os.path.join(LAKE_DELTA_UDM_DATASET_DIR, "all_datasets.dump")
        n_components = LAKE_DELTA_N_COMPONENTS
        udm_sections = LAKE_DELTA_TRAIN_SECTIONS
    else:
        raise NotImplementedError(opt.facies)

    need_datasets = [text.strip() for text in opt.dataset_type.split(",")]
    _datasets = []
    for dataset_type in need_datasets:
        if dataset_type == "udm":
            udm_dataset = UDMDataset(udm_dataset_path, sections=udm_sections, true_gsd=False, size=opt.udm_dataset_size)
            _datasets.append(udm_dataset)
        else:
            gan_dataset_path = os.path.join("./datasets/gan", opt.facies, f"{dataset_type}.pkl")
            gan_dataset = GANDataset(gan_dataset_path, opt.gan_dataset_size)
            _datasets.append(gan_dataset)
    dataset = DecomposerDataset(_datasets, device=opt.device)
    validate_size = max(min(int(len(dataset)*0.2), 2048), 128)
    validate_dataset, train_dataset = random_split(
        dataset, [validate_size, len(dataset) - validate_size],
        torch.Generator().manual_seed(42))
    validate_dataloader = DataLoader(validate_dataset, opt.batch_size, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True, drop_last=True)
    decomposer = Decomposer(n_components, N_CLASSES)
    mse_loss = torch.nn.MSELoss()
    decomposer.to(opt.device)
    mse_loss.to(opt.device)
    optimizer = torch.optim.NAdam(
        decomposer.parameters(),
        lr=opt.lr, betas=(opt.b1, opt.b2),
        weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=opt.n_batches//100, eta_min=1e-6)
    scaler = GradScaler()
    batches_done = 0
    validate_loss = {"distributions": [], "proportions": [], "components": []}
    train_loss = {"distributions": [], "proportions": [], "components": []}
    pbar = tqdm(total=opt.n_batches,
                desc=f"Training decomposers for "
                     f"{opt.facies.replace('_', ' ').capitalize()} #{opt.experiment_id}")

    def validate(model, data):
        # ---------------------
        #  Validate Decomposer
        # ---------------------
        model.eval()
        distributions, proportions, components = data
        with torch.no_grad():
            gen_proportions, gen_components = model(distributions)
            gen_distributions = torch.squeeze(gen_proportions @ gen_components, dim=1)
            loss_distributions = torch.log(mse_loss(gen_distributions, distributions))
            loss_proportions = torch.log(mse_loss(gen_proportions, proportions))
            loss_components = torch.log(mse_loss(gen_components, components))
        validate_loss["distributions"].append(loss_distributions.item())
        validate_loss["proportions"].append(loss_proportions.item())
        validate_loss["components"].append(loss_components.item())

    def train(model, data):
        # ---------------------
        #  Train Decomposer
        # ---------------------
        model.train()
        distributions, proportions, components = data
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            gen_proportions, gen_components = model(distributions)
            gen_distributions = torch.squeeze(gen_proportions @ gen_components, dim=1)
            loss_distributions = torch.log(mse_loss(gen_distributions, distributions))
            loss_proportions = torch.log(mse_loss(gen_proportions, proportions))
            loss_components = torch.log(mse_loss(gen_components, components))
            loss = (loss_distributions + loss_components + loss_proportions) / 3
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss["distributions"].append(loss_distributions.item())
        train_loss["proportions"].append(loss_proportions.item())
        train_loss["components"].append(loss_components.item())

    validate_opt = torch.compile(validate, mode="reduce-overhead")
    train_opt = torch.compile(train, mode="reduce-overhead")
    while True:
        for distributions, proportions, components in validate_dataloader:
            with torch.no_grad():
                distributions = distributions.to(opt.device)
                proportions = proportions.to(opt.device)
                components = components.to(opt.device)
            validate(decomposer, (distributions, proportions, components))
        for distributions, proportions, components in train_dataloader:
            noise_ratio = np.clip(10 ** (-4 - random.random() * -4), 1e-8, 1e-4).item()
            distributions = distributions.to(opt.device) + torch.rand(distributions.size(), device=opt.device) * noise_ratio
            proportions = proportions.to(opt.device) + torch.rand(proportions.size(), device=opt.device) * noise_ratio
            components = components.to(opt.device) + torch.rand(components.size(), device=opt.device) * noise_ratio
            train_opt(decomposer, (distributions, proportions, components))
            pbar.update(1)
            batches_done += 1
            if batches_done % opt.save_interval == 0:
                save_checkpoint(batches_done, opt, decomposer)
            if batches_done % 10000 == 0:
                with open(os.path.join(ROOT_DIR, opt.facies, str(opt.experiment_id),
                                       "loss_variations.dump"), "wb") as f:
                    loss_variations = {"validate_loss": validate_loss,
                                       "train_loss": train_loss,
                                       "batches_done": batches_done}
                    pickle.dump(loss_variations, f)
            if batches_done >= opt.n_batches:
                break
            if batches_done % 100 == 0:
                scheduler.step()
        if batches_done >= opt.n_batches:
            pbar.close()
            break
