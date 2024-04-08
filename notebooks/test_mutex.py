import random
import threading
import time

import numpy as np

import wandb

now = lambda: time.strftime("%H:%M:%S")
kv_store = {}
DONE = -1


def print_thr(*args, **kwargs):
    print(f"\t{threading.current_thread().name}", *args, **kwargs, flush=True)


def wandb_log_image(run, step, lock, task_id, log_t=None):
    global DONE
    start = now()
    target = task_id - 1

    # create a log image
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
    image = wandb.Image(pixels, caption=f"random field {step}")
    kv_store[task_id] = (step, image)

    if log_t is None:
        log_t = random.randint(0, 2)
    # simulate the (stochastic) time it takes to create the log image
    print_thr(f"TASK {task_id} sleep {log_t}s...")
    time.sleep(log_t)

    while DONE != target:
        print_thr(f"{task_id} waits for {target}")
        time.sleep(0.5)
    print_thr(f"{task_id} FOUND {target}!")

    # retrieve desired data from kv store & delete it to avoid growing it infinitely
    it, image = kv_store.pop(task_id)
    # log image to wandb server
    run.log({"random-field": image}, step=it)

    with lock:
        DONE += 1

    print_thr(f"DONE task {task_id} step {step} (start {start} end {now()})")


def train_loop_manual(run):
    logger_threads = []
    lock = threading.Lock()
    i = step = 0

    print(f"Spawning task {i} -> {now()}")
    t = threading.Thread(
        target=wandb_log_image,
        args=(run, step, lock, i),
        kwargs={"log_t": 5},
        name=f"logger-{i}",
    )
    t.start()
    logger_threads.append(t)
    i += 1
    step += 4

    print(f"Spawning task {i} -> {now()}")
    t = threading.Thread(
        target=wandb_log_image,
        args=(run, step, lock, i),
        kwargs={"log_t": 2},
        name=f"logger-{i}",
    )
    t.start()
    logger_threads.append(t)
    i += 1
    step += 4

    print(f"Spawning task {i} -> {now()}")
    t = threading.Thread(
        target=wandb_log_image,
        args=(run, step, lock, i),
        kwargs={"log_t": 4},
        name=f"logger-{i}",
    )
    t.start()
    logger_threads.append(t)
    return logger_threads


def train_loop(run, log_every: int, target_fn):
    logger_threads = []
    lock = threading.Lock()

    for i in range(5):
        # simulate expensive model training
        idle_t = random.randint(0, 3)
        print(f"it {i} sleep {idle_t} @ {now()}")
        time.sleep(idle_t)

        if i % log_every == 0:
            nth_spawn = i // log_every
            print(f"Spawning task {nth_spawn} -> {now()}")
            t = threading.Thread(
                target=target_fn,
                args=(run, i, lock, nth_spawn),
                name=f"logger-{i}",
            )
            t.start()
            logger_threads.append(t)

    return logger_threads


if __name__ == "__main__":
    start = time.time()

    run = None
    run = wandb.init(project="test_project", entity="xqz-u")

    # logger_threads = train_loop_manual(run)
    logger_threads = train_loop(run, 1, wandb_log_image)

    for t in logger_threads:
        if t.is_alive():
            print(f"Waiting on thread {t.name!r}...")
            t.join()

    print(f"took {time.time() - start:.2f}s")

    if run is not None:
        run.finish()


# NOTE RIP exception handling :')
# @dataclass
# class MultiThreadedLogger(LoggerInterface):
#     started_threads: List[Thread] = field(default_factory=list, init=False)
#     logging_step: int = field(default=0, init=False)
#     done: int = field(default=-1, init=False)
#     images_store: dict = field(default_factory=lambda: defaultdict(list), init=False)
#     lock: Lock = field(default_factory=Lock, init=False)

#     def multiple(
#         self,
#         patches: List[dict],
#         reconstructions: List[Tensor],
#         attenuations: List[Tensor],
#         iteration: int,
#         stage: str,
#     ):
#         def inner(patches, reconstructions, attenuations, stage, iteration, task_num):
#             target = task_num - 1
#             print(f"task {task_num} target {target} it {iteration}")

#             for data, scan_type, recon, attn_map in zip(
#                 patches, ScanType, reconstructions, attenuations
#             ):
#                 buffer = self.logger(
#                     data["data"],
#                     iteration,
#                     stage,
#                     scan_type.name,
#                     data["name"],
#                     masks=data["seg"],
#                     reconstructions=recon,
#                     attenuations=attn_map,
#                 )
#                 print(f"DONE {scan_type.name} task {task_num} it {iteration}")
#                 self.images_store[task_num].append(buffer)

#             while self.done != target:
#                 print(f"{task_num} {stage} looking for {target}")
#                 sleep(0.5)
#             print(f"{task_num} found {target}!")

#             buffers = self.images_store.pop(task_num)
#             for b in buffers:
#                 print(f"logging wspace {b[0][0]} it {b[0][1]}")
#                 self.logger.log_images(b)

#             with self.lock:
#                 self.done += 1
#                 print(f"task {task_num} inc done: {self.done}")
#             print(f"{now_str()}: {current_thread().name} DONE", flush=True)

#         t = Thread(
#             target=inner,
#             args=(
#                 patches,
#                 reconstructions,
#                 attenuations,
#                 stage,
#                 iteration,
#                 self.logging_step,
#             ),
#             name=f"logger-{stage}-{iteration}",
#         )
#         print(f"{now_str()}: {t.name}", flush=True)
#         self.logging_step += 1
#         t.start()
#         self.started_threads.append(t)

#     def end_hook(self):
#         for t in self.started_threads:
#             if t.is_alive():
#                 print(f"Waiting on thread {t.name!r}...", flush=True)
#                 t.join()

# def extract_slices(
#     self,
#     batches: List[dict],
#     reconstructions: List[Optional[Tensor]],
#     attenuations: List[Optional[Tensor]],
# ) -> list:
#     ret = []
#     for batch, scan_type, recon, attn_map in zip(
#         batches, ScanType, reconstructions, attenuations
#     ):
#         indexer = self.logger.create_indexer(batch["data"].shape)
#         reconstruction_slices, attenuation_slices = None, None
#         low, high = None, None
#         if recon is not None:
#             reconstruction_slices = to_CPU(recon[indexer])
#         if attn_map is not None:
#             attenuation_slices = to_CPU(attn_map[indexer])
#             sample = attn_map[indexer[0]]
#             low, high = sample.min().item(), sample.max().item()
#         ret.append(
#             [
#                 batch["data"][indexer],
#                 batch["seg"][indexer],
#                 reconstruction_slices,
#                 attenuation_slices,
#                 (low, high),
#                 scan_type.name,
#                 batch["name"][indexer[0]],
#             ]
#         )
#     return ret


# def log_slices(
#     self,
#     scan_slices: Tensor,
#     mask_slices: Optional[Tensor],
#     reconstruction_slices: Optional[Tensor],
#     attenuation_slices: Optional[Tensor],
#     attenuation_norm_bounds: Optional[Tuple[float, float]],
#     scan_type: str,
#     caption: str,
#     it: int,
#     stage: str,
# ):
#     fig, buffer = None, []
#     workspace = f"{stage}/images/{scan_type}"
#     # caption_cp =  caption

#     # show centerlines by scattering manually during training
#     if stage == "train" and mask_slices is not None:
#         ctls = mask_slices.permute(3, 0, 1, 2).to(torch.float16)
#         ctls_grid = make_grid(ctls)
#         # DHW -> HWD (yxz)
#         cart = geom.grid_to_cartesian_coords(ctls_grid.permute(1, 2, 0))
#         fig, ax = plt.subplots(figsize=self.figsize)
#         ax.scatter(
#             cart[:, 1],
#             cart[:, 0],
#             c="red",
#             s=plt.rcParams["lines.markersize"] * 0.8,
#         )
#         # caption_cp = f"{caption} {np.prod(cart.shape)}/{np.prod(masks[sample_idx].shape)} centerlines"

#     slices = self.reconstruct_sample(scan_slices)
#     fig = WandbLogger.create_grid_figure(slices, fig, **self.grid_args)
#     # buffer.append(((f"{workspace}/sample", it, fig), {"caption": caption_cp}))
#     buffer.append(((f"{workspace}/sample", it, fig), {"caption": caption}))

#     if reconstruction_slices is not None:
#         recon = self.reconstruct_optimized_sample(reconstruction_slices)
#         fig = WandbLogger.create_grid_figure(recon, **self.grid_args)
#         buffer.append(((f"{workspace}/reconstruction", it, fig), {"caption": caption}))

#     if attenuation_slices is not None and attenuation_norm_bounds is not None:
#         fig = self.create_attenuation_grid(attenuation_slices, *attenuation_norm_bounds)
#         buffer.append(((f"{workspace}/attenuation", it, fig), {"caption": caption}))

#     return buffer


# def create_attenuation_grid(
#         self, attenuation_slices: Tensor, norm_low: float, norm_high: float
#     ) -> Figure:
#         # normalize [-1, 1]->[0, 1] for colormap (min and max from entire sample)
#         attn = minmax_norm(attenuation_slices, (norm_low, norm_high))
#         attn = self.cmap(attn).squeeze().transpose(3, 0, 1, 2)
#         # add colorbar
#         fig, ax = plt.subplots(figsize=self.figsize)
#         norm = colors.Normalize(norm_low, norm_high)
#         mappable = cm.ScalarMappable(norm=norm, cmap=self.cmap)
#         cbar = fig.colorbar(mappable, ax=ax, shrink=0.8)
#         cbar.set_ticks(np.linspace(norm_low, norm_high, 5))
#         return WandbLogger.create_grid_figure(attn, fig)
