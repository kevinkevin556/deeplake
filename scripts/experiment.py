import glob
import os
import subprocess

import pandas as pd


def run_command(command):
    subprocess.run(command, shell=True, check=True)


def get_latest_checkpoint_dir(n=1):
    checkpoint_dirs = sorted(glob.glob("./checkpoints/*"), key=os.path.getmtime)
    return checkpoint_dirs[-n:] if checkpoint_dirs else None


def extract_csv_data(csv_file):
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file, index_col=0)
    return pd.DataFrame()


dice_scores = []
hausdorff_scores = []
checkpoint_dirs = []

N = 5
train = True

if train:
    for i in range(N):
        print(f"--- Iteration {i+1} ---")

        # Step 1: Train
        run_command("python train.py --config ./config/train.yml")

        # Step 2: Get latest checkpoint directory
        checkpoint_dir = get_latest_checkpoint_dir()[0]
        if not checkpoint_dir:
            print("No checkpoint directory found.")
            break

        # Step 3: Test
        run_command(f"python test.py --config ./config/test.yml --pretrained {checkpoint_dir} --evaluator all")

        # Step 4: Collect and store DSC and HD values
        dice_csv = os.path.join(checkpoint_dir, "dice.csv")
        hausdorff_csv = os.path.join(checkpoint_dir, "hausdorff.csv")
        dice_data = extract_csv_data(dice_csv)
        hausdorff_data = extract_csv_data(hausdorff_csv)
        if not dice_data.empty:
            dice_scores.append(dice_data)
        if not hausdorff_data.empty:
            hausdorff_scores.append(hausdorff_data)

        print("\n")

if not train:
    checkpoint_dirs = get_latest_checkpoint_dir(N)
    print("Num of checkpoints", len(checkpoint_dirs))
    print(checkpoint_dirs)
    for ckpt_dir in checkpoint_dirs:
        dice_csv = os.path.join(ckpt_dir, "dice.csv")
        hausdorff_csv = os.path.join(ckpt_dir, "hausdorff.csv")
        dice_data = extract_csv_data(dice_csv)
        hausdorff_data = extract_csv_data(hausdorff_csv)
        if not dice_data.empty:
            dice_scores.append(dice_data)
        if not hausdorff_data.empty:
            hausdorff_scores.append(hausdorff_data)


# Calculate average DSC and HD
print("\n\n\n")
if dice_scores:
    mean_df = pd.concat(dice_scores).groupby(level=0).mean()
    std_df = pd.concat(dice_scores).groupby(level=0).std()
    combined_df = pd.concat([mean_df, std_df], axis=1)
    original_columns = dice_scores[0].columns
    new_columns = [f"Mean_{col}" for col in original_columns] + [f"Std_{col}" for col in original_columns]
    combined_df.columns = new_columns
    combined_df = combined_df.drop(combined_df.columns[[0, 3]], axis=1)
    print("Average Dice Similarity Coefficient (DSC):\n")
    print(combined_df, "\n")

if hausdorff_scores:
    mean_df = pd.concat(hausdorff_scores).groupby(level=0).mean()
    std_df = pd.concat(hausdorff_scores).groupby(level=0).std()
    combined_df = pd.concat([mean_df, std_df], axis=1)
    original_columns = hausdorff_scores[0].columns
    new_columns = [f"Mean_{col}" for col in original_columns] + [f"Std_{col}" for col in original_columns]
    combined_df.columns = new_columns
    combined_df = combined_df.drop(combined_df.columns[[0, 3]], axis=1)
    print("Average Hausdorff Distance (HD):\n")
    print(combined_df, "\n")
