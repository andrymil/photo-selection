import glob
import os
import random


def prepare_dataset(base_dir):
    random.seed(42)

    sharp_paths = sorted(glob.glob(f"{base_dir}/CERTH/TrainingSet/Undistorted/*.*"))
    bokeh_paths = sorted(glob.glob(f"{base_dir}/CUHK/sharp/*.*"))

    all_eval_dig_paths = sorted(
        glob.glob(f"{base_dir}/CERTH/EvaluationSet/DigitalBlurSet/*.*")
    )

    eval_sharp_paths = []
    eval_blur_paths = []

    for path in all_eval_dig_paths:
        filename = os.path.basename(path)
        if filename.startswith("Original"):
            eval_sharp_paths.append(path)
        else:
            eval_blur_paths.append(path)

    ebb_bokeh_paths = sorted(glob.glob(f"{base_dir}/EBB/train/bokeh/*.*"))
    ebb_orig_paths = sorted(glob.glob(f"{base_dir}/EBB/train/original/*.*"))

    ebb_pairs = list(zip(ebb_bokeh_paths, ebb_orig_paths))

    sample_size = min(500, len(ebb_pairs))
    sampled_pairs = random.sample(ebb_pairs, sample_size)

    ebb_sampled_paths = [path for pair in sampled_pairs for path in pair]

    class_0_paths = sharp_paths + bokeh_paths + eval_sharp_paths + ebb_sampled_paths
    class_0_labels = [0] * len(class_0_paths)

    blur_nat_paths = sorted(
        glob.glob(f"{base_dir}/CERTH/TrainingSet/Naturally-Blurred/*.*")
    )
    blur_art_paths = sorted(
        glob.glob(f"{base_dir}/CERTH/TrainingSet/Artificially-Blurred/*.*")
    )
    defocused_paths = sorted(glob.glob(f"{base_dir}/Kaggle/defocused_blurred/*.*"))
    motion_blur_kaggle = sorted(glob.glob(f"{base_dir}/Kaggle/motion_blurred/*.*"))
    motion_blur = sorted(glob.glob(f"{base_dir}/CUHK/blurred/*.*"))

    class_1_paths = (
        blur_nat_paths
        + blur_art_paths
        + eval_blur_paths
        + defocused_paths
        + motion_blur_kaggle
        + motion_blur
    )
    class_1_labels = [1] * len(class_1_paths)

    all_paths = class_0_paths + class_1_paths
    all_labels = class_0_labels + class_1_labels

    print(
        f"Found {len(class_0_paths)} good (sharp) images and {len(class_1_paths)} bad (blurred) images."
    )

    if len(class_1_paths) > 0:
        weight_for_class_1 = len(class_0_paths) / len(class_1_paths)
    else:
        weight_for_class_1 = 1.0

    return all_paths, all_labels, weight_for_class_1
