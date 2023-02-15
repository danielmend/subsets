import fsspec
import io
import tarfile
import os
import tempfile
from filters import similarity_filter_CLIP_embeddings
import argparse
from braceexpand import braceexpand
import json
import open_clip
from distributed import world_info_from_env
import math
import torch
import glob

def extract_shard(shard, tempdir):
    folder = "/".join(shard.split("/")[0:-1])
    fs, output_path = fsspec.core.url_to_fs(folder)

    shard_id = shard.split("/")[-1]
    tar_bytes = io.BytesIO(fs.open(f"{output_path}/{shard_id}").read())
    with tarfile.open(fileobj=tar_bytes) as tar:
        tar.extractall(tempdir)
    return output_path, shard_id

def filter_shard(shard, filter_fn):
    with tempfile.TemporaryDirectory() as tempdir:
         output_path, shard_id = extract_shard(shard, tempdir)
         keys_to_save = filter_fn(tempdir)
         return shard_id, keys_to_save

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--shards', type=str, help="shards to filter")
    parser.add_argument('--dest', type=str, help="ids output destination")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    shards = list(braceexpand(args.shards))

    local_rank, global_rank, world_size = world_info_from_env()
    print(local_rank)
    work_size = math.ceil(len(shards) / world_size)
    ws, wf = global_rank * work_size, (global_rank + 1) * work_size
    shards = shards[ws:wf]
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)

    filter_fn = lambda tempdir: similarity_filter_CLIP_embeddings(tempdir, clip_model, device=device)


    for shard in shards:
        shard_id, keys_to_save = filter_shard(shard, filter_fn)
        with open(f'{args.dest}/{shard_id}_keep.json', 'w') as f:
             json.dump(keys_to_save, f)

if __name__ == '__main__':
    main()
