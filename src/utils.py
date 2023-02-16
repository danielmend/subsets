import fsspec
import io
import tarfile

def extract_shard(shard, tempdir):
    folder = "/".join(shard.split("/")[0:-1])
    fs, output_path = fsspec.core.url_to_fs(folder)

    shard_id = shard.split("/")[-1]
    tar_bytes = io.BytesIO(fs.open(f"{output_path}/{shard_id}").read())
    with tarfile.open(fileobj=tar_bytes) as tar:
        tar.extractall(tempdir)
    return shard_id
