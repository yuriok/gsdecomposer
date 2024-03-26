import argparse
import os
import time

from gsdecomposer.decomposer.train import ROOT_DIR


train_commands = [
    # loess
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm --udm_dataset_size 512 --experiment_id 1",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm --udm_dataset_size 1024 --experiment_id 2",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm --udm_dataset_size 2048 --experiment_id 3",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm --udm_dataset_size 4096 --experiment_id 4",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 12288 --experiment_id 5",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 24576 --experiment_id 6",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 49152 --experiment_id 7",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 98304 --experiment_id 8",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 196608 --experiment_id 9",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 393216 --experiment_id 10",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 2048 --experiment_id 11",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 4096 --experiment_id 12",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 8192 --experiment_id 13",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 16384 --experiment_id 14",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 32768 --experiment_id 15",
    "python ./gsdecomposer/decomposer/train.py --facies loess --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 65536 --experiment_id 16",

    # fluvial
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm --udm_dataset_size 512 --experiment_id 1",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm --udm_dataset_size 1024 --experiment_id 2",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm --udm_dataset_size 2048 --experiment_id 3",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm --udm_dataset_size 4096 --experiment_id 4",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 12288 --experiment_id 5",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 24576 --experiment_id 6",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 49152 --experiment_id 7",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 98304 --experiment_id 8",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 196608 --experiment_id 9",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 393216 --experiment_id 10",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 2048 --experiment_id 11",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 4096 --experiment_id 12",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 8192 --experiment_id 13",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 16384 --experiment_id 14",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 32768 --experiment_id 15",
    "python ./gsdecomposer/decomposer/train.py --facies fluvial --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 65536 --experiment_id 16",
    # lake_delta
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm --udm_dataset_size 512 --experiment_id 1",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm --udm_dataset_size 1024 --experiment_id 2",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm --udm_dataset_size 2048 --experiment_id 3",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm --udm_dataset_size 4096 --experiment_id 4",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 12288 --experiment_id 5",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 24576 --experiment_id 6",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 49152 --experiment_id 7",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 98304 --experiment_id 8",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 196608 --experiment_id 9",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 393216 --experiment_id 10",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 2048 --experiment_id 11",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 4096 --experiment_id 12",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 8192 --experiment_id 13",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 16384 --experiment_id 14",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 32768 --experiment_id 15",
    "python ./gsdecomposer/decomposer/train.py --facies lake_delta --lr 0.0004 --dataset_type udm,mlp_gan,mlp_wgan,mlp_sngan,conv_gan,conv_wgan,conv_sngan --udm_dataset_size 4096 --gan_dataset_size 65536 --experiment_id 16",
]

def all_finished():
    last_batch = 500000
    train_facies = ["loess", "fluvial", "lake_delta"]
    experiment_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    for facies in train_facies:
        for experiment_id in experiment_ids:
            last_batch_path = os.path.join(
                ROOT_DIR, facies, str(experiment_id),
                "checkpoints", f"{last_batch}.pkl")
            if not os.path.exists(last_batch_path):
                print(f"The training for "
                      f"{facies.replace('_', ' ').capitalize()} #{experiment_id}"
                      f"has not been finished.")
                return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("-s", "--server", action="store_true")
    opt = parser.parse_args()

    for train_command in train_commands[opt.job_index::opt.n_jobs]:
        os.system(train_command + f" --device {opt.device}")
    time.sleep(60.0)
    if opt.server and all_finished():
        os.system("chmod +x ./upload.sh")
        os.system("./upload.sh")

# nohup python gsdecomposer/decomposer/run_tasks.py --n_jobs 6 --job_index 0 -s > task_0.log 2>&1 &
# nohup python gsdecomposer/decomposer/run_tasks.py --n_jobs 6 --job_index 1 -s > task_1.log 2>&1 &
# nohup python gsdecomposer/decomposer/run_tasks.py --n_jobs 6 --job_index 2 -s > task_2.log 2>&1 &
# nohup python gsdecomposer/decomposer/run_tasks.py --n_jobs 6 --job_index 3 -s > task_3.log 2>&1 &
# nohup python gsdecomposer/decomposer/run_tasks.py --n_jobs 6 --job_index 4 -s > task_4.log 2>&1 &
# nohup python gsdecomposer/decomposer/run_tasks.py --n_jobs 6 --job_index 5 -s > task_5.log 2>&1 &
