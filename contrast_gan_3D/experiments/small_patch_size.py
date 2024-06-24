from contrast_gan_3D.alias import ScanType
from contrast_gan_3D.experiments.basic_conf import train_transform_args

train_patch_size = (128, 128, 32)  # decrease total number of voxel by factor 4
train_transform_args["patch_size"] = train_patch_size

train_batch_size = {
    v.value: b
    for v, b in [(ScanType.OPT, 40), (ScanType.LOW, 20), (ScanType.HIGH, 20)]
    # v.value: b for v, b in [(ScanType.OPT, 6), (ScanType.LOW, 3), (ScanType.HIGH, 3)]
}
