from writer import WebDatasetWriter

def subsample_shard(shard, tempdir, keys, writer):
    shard_id = int(shard.split('.tar')[0].split('/')[-1])
    writer.create_shard(shard_id=shard_id)
    for key in keys:
        with open(tempdir + '/' + key + '.json', 'rb') as f:
             meta = json.load(f)
        with open(tempdir + '/' + key + '.txt', 'rb') as f:
             txt = str(f.read())
             meta['caption'] = txt
             embeddings = np.load(tempdir + '/' + key + '.npy')
             writer.write(embeddings, key, meta)
