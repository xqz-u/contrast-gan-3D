import sys

sys.path.append("..")

from batchgenerators.dataloading.multi_threaded_augmenter import \
    MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import \
    SingleThreadedAugmenter

from contrast_gan_3D.constants import TRAIN_PATCH_SIZE
from contrast_gan_3D.data import utils as dset_utils
from contrast_gan_3D.data.CCTADataLoader import CCTADataLoader

# from pathlib import Path
#
# import numpy as np
# from batchgenerators.dataloading.single_threaded_augmenter import \
#     SingleThreadedAugmenter
# from batchgenerators.transforms.abstract_transforms import Compose
# from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
#
#
# from batchviewer import view_batch
#
# from contrast_gan_3D.config import DEFAULT_SEED
# from contrast_gan_3D.data.CCTADataLoader import CCTADataLoader
# from contrast_gan_3D.utils import io_utils
#
# # data_dirs = ["MMWHS/ct_train", "MMWHS/ct_test", "ASOCA_Philips/images"]
# # mmwhs_train, mmwhs_test, asoca = [], [], []
# # for d, store in zip(data_dirs, [mmwhs_train, mmwhs_test, asoca]):
# #     paths = sorted((Path("/home/marco/data") / d).glob("*.h5"))
# #     for p in paths:
# #         img, meta, f = io_utils.load_h5_image(p)
# #         if len(f["ccta"]["centerlines"].shape) < 2:
# #             print(io_utils.stem(p))
# #         else:
# #             store.append(p)
# #         f.close()
# # valid_paths = mmwhs_train + mmwhs_test + asoca
# # print(len(valid_paths))
# # len(mmwhs_train), len(mmwhs_test), len(asoca)
#
#
# bs = 4
# # ps = (512, 512, 128)
# ps = (128, 128, 128)
# # loader = CCTADataLoader(valid_paths, batch_size=bs)
# # loader = CCTADataLoader(mmwhs_train, batch_size=bs)
# # loader = CCTADataLoader(asoca, batch_size=bs)
# loader = CCTADataLoader(
#     ["/home/marco/data/ASOCA_Philips/images/ASOCA-000.h5"], batch_size=bs, patch_size=ps
# )
#
# rot_angles = (0, np.pi)
# print(rot_angles)
#
# # only rotation
# transforms = [
#     SpatialTransform_2(
#         loader.patch_size,
#         random_crop=False,
#         do_rotation=True,
#         do_scale=False,
#         do_elastic_deform=False,
#         angle_x=rot_angles,
#         angle_y=rot_angles,
#         angle_z=rot_angles,
#         p_rot_per_sample=1,
#     )
# ]
# transforms = Compose(transforms)
# gen = SingleThreadedAugmenter(loader, transforms)
#
# batch = next(gen)
# print(list(batch))
# print(batch["patch_origin"])
# print(batch["names"])
#
# for idx in range(len(batch["data"])):
#     print(batch["names"][idx], batch["patch_origin"][idx])
#     view_batch(batch["data"][idx].swapaxes(1, -1), batch["seg"][idx].swapaxes(1, -1))


GENERATOR_CLASS = SingleThreadedAugmenter
# NOTE it seems that this generator goes through the data one whole time per thread!
GENERATOR_CLASS = MultiThreadedAugmenter
args = (3,)

loaders = CCTADataLoader.create_from_cval(
    3, 3, TRAIN_PATCH_SIZE, -1, "/home/marco/data/ostia_final.xlsx"
)
cval_idx = 0
# loaders = {"train": loaders["train"][cval_idx], "val": loaders["val"][cval_idx]}
# print(len(loaders["val"]._data))

train_gen = GENERATOR_CLASS(
    loaders["train"], dset_utils.make_train_transforms(TRAIN_PATCH_SIZE), *args
)
# val_gen = GENERATOR_CLASS(loaders["val"], None, *args)

label_counts = {}

train_transform = dset_utils.make_train_transforms(TRAIN_PATCH_SIZE)
for j, train_loader in enumerate(loaders["train"]):
    train_gen = SingleThreadedAugmenter(train_loader, train_transform)
    for i in range(100):
        batch = next(train_gen)
        # print(batch["data"].shape, batch["seg"].shape)
        # print(batch["labels"])
        for lab in batch["labels"]:
            if j not in label_counts:
                label_counts[j] = {-1: 0, 0: 0, 1: 0}
            label_counts[j][lab] += 1
        # for m in batch["meta"]:
        #     p = m["path"]
        #     print(p)
        print(f"{i} -----")
    print(label_counts)
