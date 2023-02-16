import os
import json

import fsspec
import numpy as np
import webdataset as wds

from io import BytesIO

class WebDatasetWriter:
    """Writes output in WebDataset format."""

    def __init__(self, output_folder, oom_shard_count, shard_id=0):
        self.output_folder = output_folder
        self.oom_shard_count = oom_shard_count
        self.shard_id = shard_id

        self.count = 0

        self.tarwriter = None
        self.tar_fd = None

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

    def write(self, data):
        """write sample to current shard."""
        key = str(data.pop('key'))

        sample = {"__key__": key}
        for ext, file in data.items():
            with open(file, "rb") as stream:
                file_data = stream.read()
            sample[ext] = file_data

        self.tarwriter.write(sample)
        self.count += 1

    def close(self):
        if self.tarwriter is not None:
            self.tarwriter.close()
            self.tar_fd.close()
