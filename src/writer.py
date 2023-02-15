import os
import json

import fsspec
import numpy as np
import webdataset as wds

from io import BytesIO

class WebDatasetWriter:
    """Writes output in WebDataset format."""

    def __init__(self, output_folder, oom_shard_count, encode_format, maxcount=10000, shard_id=0):
        self.output_folder = output_folder
        self.oom_shard_count = oom_shard_count
        self.encode_format = encode_format
        self.maxcount = maxcount
        self.shard_id = shard_id

        self.count = 0

        self.tarwriter = None
        self.tar_fd = None

        # self.create_shard()

    def create_shard(self, shard_id=None):
        """create new shard in sequential order."""
        self.close()
        if shard_id is not None:
            self.shard_id = shard_id
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=self.shard_id, oom_shard_count=self.oom_shard_count
        )
        fs, output_path = fsspec.core.url_to_fs(self.output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)

    def write(self, arr, key, metadata=None):
        """write sample to current shard."""
        key = str(key)
        if self.count >= self.maxcount:
            self.shard_id += 1
            self.count = 0
            self.create_shard()

        sample = {"__key__": key, self.encode_format: arr}
        if metadata is not None:
            if "caption" in metadata:
                sample["txt"] = str(metadata.pop("caption"))
            if len(metadata) > 0:
                sample["json"] = json.dumps(metadata, indent=4)

        self.tarwriter.write(sample)
        self.count += 1

    def close(self):
        if self.tarwriter is not None:
            self.tarwriter.close()
            self.tar_fd.close()
