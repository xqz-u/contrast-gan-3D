import sys

sys.path.append("..")

import pandas as pd
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchviewer import view_batch

from contrast_gan_3D.constants import TRAIN_PATCH_SIZE
from contrast_gan_3D.data.CCTADataLoader3D import CCTADataLoader3D
from contrast_gan_3D.utils import geometry as geom

sheet = pd.read_excel("/home/marco/data/ostia_final.xlsx")
paths, labels = sheet["path"].values, sheet["label"].values
# paths, labels = (['/home/marco/data/ASOCA_Philips/images/ASOCA-002.h5'], [1])

bs = 4
# bs = 1
ps = TRAIN_PATCH_SIZE
# ps = (512, 512, 128)
loader = CCTADataLoader3D(paths, ps, bs, shuffle=True, infinite=True, max_HU_diff=600)

train_transform_args = {
    "patch_size": ps,
    "random_crop": False,
    # deformation
    "do_elastic_deform": True,
    "deformation_scale": (0, 0.25),  # default
    "p_el_per_sample": 0.1,
    # scaling
    "do_scale": True,
    "scale": (0.7, 1.4),  # default
    "p_scale_per_sample": 0.2,
    # rotation
    "do_rotation": True,
    **{
        f"angle_{ax}": (-geom.deg_to_radians(30), geom.deg_to_radians(30))
        for ax in list("xyz")
    },
    "p_rot_per_sample": 0.2,
}
transforms = Compose(
    [
        SpatialTransform_2(**train_transform_args),
        NumpyToTensor(keys=["data"]),
        NumpyToTensor(keys=["seg"], cast_to="bool"),
    ]
)
# transforms = None
augmenter = SingleThreadedAugmenter

gen = augmenter(loader, transforms)
print(len(gen.data_loader))

batch = next(gen)
print(batch["data"].shape, batch["data"].dtype, batch["seg"].shape, batch["seg"].dtype)
print(batch["meta"])

for idx in range(len(batch["data"])):
    data, seg = batch["data"][idx], batch["seg"][idx]
    print(batch["name"][idx])
    # CWHD -> CDWH for **axial** view
    view_batch(data.permute((0, 3, 1, 2)), seg.permute((0, 3, 1, 2)))
