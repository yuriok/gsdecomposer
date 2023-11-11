import argparse
import os
import random
import time

from gsdecomposer.gan.train import ROOT_DIR


train_commands = [
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type mlp_gan --facies loess",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --clip_value 0.1 --network_type mlp_wgan --facies loess",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type mlp_sngan --facies loess",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type conv_gan --facies loess",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --clip_value 0.1 --network_type conv_wgan --facies loess",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type conv_sngan --facies loess",

    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type mlp_gan --facies fluvial",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --clip_value 0.1 --network_type mlp_wgan --facies fluvial",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type mlp_sngan --facies fluvial",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type conv_gan --facies fluvial",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --clip_value 0.1 --network_type conv_wgan --facies fluvial",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type conv_sngan --facies fluvial",

    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type mlp_gan --facies lake_delta",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --clip_value 0.1 --network_type mlp_wgan --facies lake_delta",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type mlp_sngan --facies lake_delta",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type conv_gan --facies lake_delta",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --clip_value 0.1 --network_type conv_wgan --facies lake_delta",
    "python ./gsdecomposer/gan/train.py --lr_G 0.00004 --lr_D 0.00004 --network_type conv_sngan --facies lake_delta",
]

check_commands = [
    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_gan --facies loess",
    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_wgan --facies loess",
    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_sngan --facies loess",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_gan --facies loess",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_wgan --facies loess",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_sngan --facies loess",

    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_gan --facies fluvial",
    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_wgan --facies fluvial",
    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_sngan --facies fluvial",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_gan --facies fluvial",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_wgan --facies fluvial",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_sngan --facies fluvial",

    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_gan --facies lake_delta",
    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_wgan --facies lake_delta",
    "python ./gsdecomposer/gan/check_detailed.py --network_type mlp_sngan --facies lake_delta",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_gan --facies lake_delta",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_wgan --facies lake_delta",
    "python ./gsdecomposer/gan/check_detailed.py --network_type conv_sngan --facies lake_delta",
]


def all_finished():
    last_batch = 500000
    train_facies = ["loess", "fluvial", "lake_delta"]
    train_networks = ["mlp_gan", "mlp_wgan", "mlp_sngan", "conv_gan", "conv_wgan", "conv_sngan"]
    for facies in train_facies:
        for network_type in train_networks:
            last_batch_path = os.path.join(ROOT_DIR, facies, network_type, "checkpoints", f"{last_batch}.pkl")
            if not os.path.exists(last_batch_path):
                print(f"The training of {network_type.replace('_', ' ').upper()}s for "
                      f"{facies.replace('_', ' ').capitalize()} "
                      f"has not been finished.")
                return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("-c", "--check", action="store_true")
    parser.add_argument("-f", "--fast", action="store_true")
    parser.add_argument("-n", "--n_cpus", type=int, default=8)
    opt = parser.parse_args()
    if opt.train:
        command_index = list(range(len(train_commands)))
        # random.shuffle(command_index)
        for index in command_index[opt.job_index::opt.n_jobs]:
            os.system(train_commands[index] + f" --device {opt.device}")
        time.sleep(60.0)
        if opt.server and all_finished():
            os.system("chmod +x ./upload.sh")
            os.system("./upload.sh")

    if opt.check:
        for command in check_commands:
            os.system(command + f" --n_cpus {opt.n_cpus}{' --fast ' if opt.fast else ''}")


# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 0 > task_0.log 2>&1 &
# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 1 > task_1.log 2>&1 &
# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 2 > task_2.log 2>&1 &
# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 3 > task_3.log 2>&1 &
# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 4 > task_4.log 2>&1 &
# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 5 > task_5.log 2>&1 &
# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 6 > task_6.log 2>&1 &
# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 7 > task_7.log 2>&1 &
# nohup python gsdecomposer/gan/run_tasks.py --train --n_jobs 9 --job_index 8 > task_8.log 2>&1 &
