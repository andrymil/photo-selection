import glob
import os


def prepare_dataset(base_dir):
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

    class_0_paths = sharp_paths + bokeh_paths + eval_sharp_paths
    class_0_labels = [0] * len(class_0_paths)

    blur_nat_paths = sorted(
        glob.glob(f"{base_dir}/CERTH/TrainingSet/Naturally-Blurred/*.*")
    )
    blur_art_paths = sorted(
        glob.glob(f"{base_dir}/CERTH/TrainingSet/Artificially-Blurred/*.*")
    )
    motion_blur = sorted(glob.glob(f"{base_dir}/CUHK/blurred/*.*"))

    class_1_paths = blur_nat_paths + blur_art_paths + eval_blur_paths + motion_blur
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
