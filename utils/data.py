import glob


def prepare_dataset(base_dir):
    sharp_paths = glob.glob(f"{base_dir}/CERTH/TrainingSet/Undistorted/*.*")
    bokeh_paths = glob.glob(f"{base_dir}/CUHK/sharp/*.*")
    class_0_paths = sharp_paths + bokeh_paths
    class_0_labels = [0] * len(class_0_paths)

    blur_nat_paths = glob.glob(f"{base_dir}/CERTH/TrainingSet/Naturally-Blurred/*.*")
    blur_art_paths = glob.glob(f"{base_dir}/CERTH/TrainingSet/Artificially-Blurred/*.*")
    eval_blur_nat_paths = glob.glob(
        f"{base_dir}/CERTH/EvaluationSet/NaturalBlurSet/*.*"
    )
    eval_blur_dig_paths = glob.glob(
        f"{base_dir}/CERTH/EvaluationSet/DigitalBlurSet/*.*"
    )
    motion_blur = glob.glob(f"{base_dir}/CUHK/blurred/*.*")
    class_1_paths = (
        blur_nat_paths
        + blur_art_paths
        + eval_blur_nat_paths
        + eval_blur_dig_paths
        + motion_blur
    )
    class_1_labels = [1] * len(class_1_paths)

    all_paths = class_0_paths + class_1_paths
    all_labels = class_0_labels + class_1_labels

    print(f"Found {len(class_0_paths)} good images and {len(class_1_paths)} bad.")

    weight_for_class_1 = len(class_0_paths) / len(class_1_paths)

    return all_paths, all_labels, weight_for_class_1
