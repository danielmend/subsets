from writer import WebDatasetWriter
import os
import json
import tempfile
from utils import extract_shard
import argparse
from braceexpand import braceexpand

def group_shard_by_samples(tempdir, keys):
    extensions = set()
    for file in os.listdir(tempdir):
        key, ext = os.path.splitext(file)
        extensions.add(ext[1:])

    samples = []
    for key in keys:
        s = {}
        s['key'] = key
        for ext in extensions:
            s[ext] = f'{tempdir}/{key}.{ext}'
        samples.append(s)

    return samples

def subsample_shard(shard, tempdir, keys, writer):
    shard_id = int(shard.split('.tar')[0].split('/')[-1])
    writer.create_shard(shard_id=shard_id)
    samples = group_shard_by_samples(tempdir, keys)
    print(samples[0])
    for sample in samples:
        writer.write(sample)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--shards', type=str, help="shards to subsample")
    parser.add_argument('--keys_dir', type=str, help="directory with json files containing info about which keys to keep in each shard")
    parser.add_argument('--dest', type=str, help="subset destination")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    shards = list(braceexpand(args.shards))
    writer = WebDatasetWriter(args.dest, 9)
    for shard in shards:
        with tempfile.TemporaryDirectory() as tempdir:
            shard_id = extract_shard(shard, tempdir)
            shard_id = shard_id.split('.tar')[0]
            with open(f'{args.keys_dir}{shard_id}_keep.json', 'rb') as f:
                keys_to_keep = json.load(f)
            subsample_shard(shard, tempdir, keys_to_keep, writer)

if __name__ == "__main__":
    main()
