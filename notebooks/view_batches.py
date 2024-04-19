import sys

sys.path.append("..")

from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchviewer import view_batch

from contrast_gan_3D.data.CCTADataLoader import CCTADataLoader

# from contrast_gan_3D.experiments.conf_2D import *
from contrast_gan_3D.experiments.basic_conf import *

# sheet = pd.read_excel("/home/marco/data/ostia_final.xlsx")
# paths = sheet["path"].values
paths = ["/home/marco/data/ASOCA_Philips/images/ASOCA-006.h5"]
# paths, labels = (['/home/marco/data/ASOCA_Philips/images/ASOCA-002.h5'], [1])

bs = 4
# ps = TRAIN_PATCH_SIZE
# ps = (512, 512, 128)
ps = train_patch_size
# ps = val_patch_size
train_transform_args["patch_size"] = ps
loader = CCTADataLoader(paths, ps, bs, rng, scaler=scaler, shuffle=True, infinite=True)
# loader = CCTADataLoader(paths, ps, bs, rng, shuffle=True, infinite=True)

transforms = Compose(
    [
        SpatialTransform_2(**train_transform_args),
        # MirrorTransform(axes=(0, 1), p_per_sample=0.5),
        NumpyToTensor(keys=["data"]),
        NumpyToTensor(keys=["seg"], cast_to="bool"),
    ]
)
augmenter = SingleThreadedAugmenter

gen = augmenter(loader, transforms)
print(len(gen.data_loader))

batch = next(gen)
print(batch["data"].shape, batch["data"].dtype, batch["seg"].shape, batch["seg"].dtype)
for b in batch["data"]:
    print(b.min(), b.max())

axes = (0, 3, 2, 1) if len(batch["data"].shape) > 4 else (0, 2, 1)
for idx in range(len(batch["data"])):
    data, seg = batch["data"][idx], batch["seg"][idx]
    print(batch["name"][idx])
    # CHW(D) -> C(D)WH for **axial** view
    view_batch(data.permute(axes), seg.permute(axes))
