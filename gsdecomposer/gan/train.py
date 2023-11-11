import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from gsdecomposer import N_CLASSES, setup_seed
from gsdecomposer.udm.dataset import UDMDataset
from gsdecomposer.udm.loess import UDM_DATASET_DIR as LOESS_UDM_DATASET_DIR, \
    TRAIN_SECTIONS as LOESS_TRAIN_SECTIONS
from gsdecomposer.udm.fluvial import UDM_DATASET_DIR as FLUVIAL_UDM_DATASET_DIR, \
    TRAIN_SECTIONS as FLUVIAL_TRAIN_SECTIONS
from gsdecomposer.udm.lake_delta import UDM_DATASET_DIR as LAKE_DELTA_UDM_DATASET_DIR, \
    TRAIN_SECTIONS as LAKE_DELTA_TRAIN_SECTIONS
from gsdecomposer.gan.mlp import MLPGenerator, MLPDiscriminator
from gsdecomposer.gan.conv import ConvGenerator, ConvDiscriminator


ROOT_DIR = os.path.abspath("../results/gan")


def save_checkpoint(
        batches_done: int, options: argparse.Namespace, generator_states: dict,
        latent_dim: int, n_components: int, n_classes: int):
    checkpoint = {"options": options,
                  "generator_states": generator_states,
                  "latent_dim": latent_dim,
                  "n_components": n_components,
                  "n_classes": n_classes}
    os.makedirs(os.path.join(ROOT_DIR, options.facies, options.network_type,
                             "checkpoints"), exist_ok=True)
    torch.save(checkpoint, os.path.join(ROOT_DIR, options.facies, options.network_type,
                                        "checkpoints", f"{batches_done}.pkl"))


if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 1
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda",
                        help="the device used to train")
    parser.add_argument("--n_batches", type=int, default=500000,
                        help="number of batches of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="dimensionality of the latent space")
    parser.add_argument("--ngf", type=int, default=64,
                        help="number of generator features")
    parser.add_argument("--ndf", type=int, default=64,
                        help="number of discriminator features")
    parser.add_argument("--network_type", type=str, default="mlp_gan",
                        help="the network type of the generator and discriminator "
                             "(mlp_gan, mlp_wgan, mlp_sngan, "
                             "conv_gan, conv_wgan, conv_sngan)")
    parser.add_argument("--lr_G", type=float, default=5e-5,
                        help="adam: learning rate of generator")
    parser.add_argument("--lr_D", type=float, default=5e-5,
                        help="adam: learning rate of discriminator")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of second order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="adam: weight decay")
    parser.add_argument("--clip_value", type=float, default=-1.0,
                        help="lower and upper clip value for weights")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="batch interval of saving checkpoints")
    parser.add_argument("--facies", type=str, default="loess",
                        help="the sedimentary facies to train")
    parser.add_argument("--no_compile", action="store_true",
                        help="do not use torch.compile for better performance")
    opt = parser.parse_args()
    setup_seed(42)
    torch.cuda.set_device(0 if opt.device == "cuda" else opt.device)
    if opt.facies == "loess":
        dataset_path = os.path.join(LOESS_UDM_DATASET_DIR, "all_datasets.dump")
        sections = LOESS_TRAIN_SECTIONS
    elif opt.facies == "fluvial":
        dataset_path = os.path.join(FLUVIAL_UDM_DATASET_DIR, "all_datasets.dump")
        sections = FLUVIAL_TRAIN_SECTIONS
    elif opt.facies == "lake_delta":
        dataset_path = os.path.join(LAKE_DELTA_UDM_DATASET_DIR, "all_datasets.dump")
        sections = LAKE_DELTA_TRAIN_SECTIONS
    else:
        raise NotImplementedError(opt.facies)

    if opt.network_type in ("mlp_gan", "mlp_wgan", "mlp_sngan"):
        Generator = MLPGenerator
        Discriminator = MLPDiscriminator
    elif opt.network_type in ("conv_gan", "conv_wgan", "conv_sngan"):
        Generator = ConvGenerator
        Discriminator = ConvDiscriminator
    else:
        raise NotImplementedError(opt.network_type)
    dataset = UDMDataset(dataset_path, sections=sections, true_gsd=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=True, drop_last=True)
    generator = Generator(opt.latent_dim, dataset.n_components, N_CLASSES, opt.ngf)
    spectral = opt.network_type in ("mlp_sngan", "conv_sngan")
    discriminator = Discriminator(dataset.n_components, N_CLASSES, opt.ndf, spectral=spectral)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    generator.to(opt.device)
    discriminator.to(opt.device)
    bce_loss.to(opt.device)
    if opt.network_type in ("mlp_wgan", "conv_wgan"):
        optimizer_G = torch.optim.RMSprop(
            generator.parameters(), lr=opt.lr_G,
            weight_decay=opt.weight_decay)
        optimizer_D = torch.optim.RMSprop(
            discriminator.parameters(), lr=opt.lr_D,
            weight_decay=opt.weight_decay)
    else:
        optimizer_G = torch.optim.NAdam(
            generator.parameters(),
            lr=opt.lr_G, betas=(opt.b1, opt.b2),
            weight_decay=opt.weight_decay)
        optimizer_D = torch.optim.NAdam(
            discriminator.parameters(),
            lr=opt.lr_D, betas=(opt.b1, opt.b2),
            weight_decay=opt.weight_decay)
    scaler = GradScaler()
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_G, T_0=opt.n_batches//100, T_mult=1, eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_D, T_0=opt.n_batches//100, T_mult=1, eta_min=1e-6)
    batches_done = 0
    g_loss_series = []
    d_loss_series = []
    pbar = tqdm(total=opt.n_batches,
                desc=f"Training {opt.network_type.replace('_', ' ').upper()}s for "
                     f"{opt.facies.replace('_', ' ').capitalize()}")

    def train(g, d, data):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        distributions, proportions, components, z, real, fake,\
            noise_distributions, noise_proportions, noise_components = data
        optimizer_D.zero_grad(set_to_none=True)
        with autocast():
            distributions += noise_distributions
            proportions += noise_proportions
            components += noise_components
            gen_proportions, gen_components = g(z)
            gen_distributions = torch.squeeze(gen_proportions @ gen_components, dim=1)
            gen_distributions = gen_distributions.detach() + noise_distributions
            gen_proportions = gen_proportions.detach() + noise_proportions
            gen_components = gen_components.detach() + noise_components
            if opt.network_type in ("mlp_wgan", "mlp_sngan", "conv_wgan", "conv_sngan"):
                d_loss = -torch.mean(d(distributions, proportions, components)) + \
                         torch.mean(d(gen_distributions, gen_proportions, gen_components))
            else:
                real_loss = bce_loss(d(distributions, proportions, components), real)
                fake_loss = bce_loss(d(gen_distributions, gen_proportions, gen_components), fake)
                d_loss = (real_loss + fake_loss) / 2
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()
        if opt.clip_value != -1.0:
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad(set_to_none=True)
        with autocast():
            gen_proportions, gen_components = g(z)
            gen_distributions = torch.squeeze(gen_proportions @ gen_components, dim=1)
            gen_distributions = gen_distributions + noise_distributions
            gen_proportions = gen_proportions + noise_proportions
            gen_components = gen_components + noise_components
            if opt.network_type in ("mlp_wgan", "mlp_sngan", "conv_wgan", "conv_sngan"):
                g_loss = -torch.mean(d(gen_distributions, gen_proportions, gen_components))
            else:
                g_loss = bce_loss(d(gen_distributions, gen_proportions, gen_components), real)
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()
        g_loss_series.append(g_loss.item())
        d_loss_series.append(d_loss.item())

    if not opt.no_compile:
        train = torch.compile(train, mode="reduce-overhead")
    while True:
        for distributions, proportions, components in dataloader:
            batch_size = distributions.size()[0]
            noise_ratio = np.clip(10 ** (-4 - batches_done / 1e5 * -4), 1e-8, 1e-4).item()
            distributions = distributions.to(opt.device)
            proportions = proportions.to(opt.device)
            components = components.to(opt.device)
            noise_distributions = torch.rand(distributions.size(), device=opt.device) * noise_ratio
            noise_proportions = torch.rand(proportions.size(), device=opt.device) * noise_ratio
            noise_components = torch.rand(components.size(), device=opt.device) * noise_ratio
            z = torch.randn((batch_size, opt.latent_dim), device=opt.device)
            real = torch.randn((batch_size, 1), device=opt.device) * 3e-2 + 0.20
            fake = torch.randn((batch_size, 1), device=opt.device) * 3e-2 + 0.80

            train(generator, discriminator,
                  (distributions, proportions, components, z, real, fake,
                   noise_distributions, noise_proportions, noise_components))
            pbar.update(1)
            batches_done += 1
            if batches_done % opt.save_interval == 0:
                save_checkpoint(batches_done, opt, generator.state_dict(),
                                opt.latent_dim, dataset.n_components, N_CLASSES)
            if batches_done % 10000 == 0:
                with open(os.path.join(ROOT_DIR, opt.facies, opt.network_type,
                                       "loss_variations.dump"), "wb") as f:
                    loss_variations = {
                        "g_loss": np.array(g_loss_series),
                        "d_loss": np.array(d_loss_series),
                        "batches_done": batches_done}
                    pickle.dump(loss_variations, f)
            if batches_done % 100 == 0:
                scheduler_G.step()
                scheduler_D.step()
            if batches_done >= opt.n_batches:
                break
        if batches_done >= opt.n_batches:
            pbar.close()
            break
