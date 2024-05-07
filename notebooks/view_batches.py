import sys

sys.path.append("..")

from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchviewer import view_batch

from contrast_gan_3D.data.CCTADataLoader import CCTADataLoader
from contrast_gan_3D.experiments.conf_2D import *

# from contrast_gan_3D.experiments.basic_conf import *

# sheet = pd.read_excel("/home/marco/data/ostia_final.xlsx")
# paths = sheet["path"].values
paths = [
    "/home/marco/data/preproc/ASOCA_Philips/ASOCA-006",
    "/home/marco/data/preproc/ASOCA_Philips/ASOCA-009",
]

bs = 4
# ps = TRAIN_PATCH_SIZE
# ps = (512, 512, 128)
ps = train_patch_size
# ps = val_patch_size
train_transform_args["patch_size"] = ps
loader = CCTADataLoader(paths, ps, bs, rng, scaler=scaler, shuffle=True, infinite=True)

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

permute_axes = (0, 3, 1, 2)
for idx in range(len(batch["data"])):
    data, seg = batch["data"][idx], batch["seg"][idx]
    print(batch["name"][idx])
    if len(batch["data"].shape) > 4:
        # CWHD -> CDWH for **axial** view
        data, seg = data.permute(permute_axes), seg.permute(permute_axes)
    view_batch(data, seg)
