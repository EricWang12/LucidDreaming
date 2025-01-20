import os
import torch.multiprocessing as mlp

def copy_and_unzip_imgs_320(idx):
    os.system(f"cp /storage-main/dmv_nocs_240207/hdri-800k-{idx:02d}.zip /host/mnt/data/render_imgs_objaverse_320/")
    os.system(f"cd /host/mnt/data/render_imgs_objaverse_320/ && unzip hdri-800k-{idx:02d}.zip")
    os.system(f"cd /host/mnt/data/render_imgs_objaverse_320/ && rm hdri-800k-{idx:02d}.zip")


print("copy_and_unzip_imgs_320 begin")
numThreads = 100
procs = []
for proc_index in range(numThreads):
    proc = mlp.Process(target=copy_and_unzip_imgs_320, args=(proc_index,))
    proc.start()
    procs.append(proc)

for i in range(len(procs)):
    procs[i].join()

print("copy_and_unzip_imgs_320 done")

print("copy_and_unzip_imgs_320 begin")
numThreads = 100
procs = []
for proc_index in range(numThreads):
    proc = mlp.Process(target=copy_and_unzip_imgs_320, args=(proc_index + 100,))
    proc.start()
    procs.append(proc)

for i in range(len(procs)):
    procs[i].join()

print("copy_and_unzip_imgs_320 done")