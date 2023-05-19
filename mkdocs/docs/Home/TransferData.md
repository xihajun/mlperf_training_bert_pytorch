## GPU Provider Data Transfer Guide

Note that `/home/ubuntu/` is the provider home directory. You can change this to the one you have.

### Step 1: Install necessary tools on `chai` (data source)

On your local machine (Chai), install `parallel`:

```bash
apt-get install parallel
```


### Step 2.0: Move data from Chai to GPU provider

Use `rsync` and `parallel` to move data from Chai to the GPU provider:

```bash
# Move training data
find /ssd/hdf5_4320_shards_varlength/ -type f | parallel -j4 --eta 'rsync -az --info=progress2 {} ubuntu@GPU_PROVIDER_IP:/home/ubuntu/trainingdata'

# Move phase1 data
find /data/bert/phase1/ -type f | parallel -j4 --eta 'rsync -az --info=progress2 {} ubuntu@GPU_PROVIDER_IP:/home/ubuntu/phase1'
```

Replace `GPU_PROVIDER_IP` with the IP address of your GPU provider machine.

### Step 2.1: Move data from cloud storage to GPU provider

If you have data on a cloud storage machine, you can move it to the GPU provider as follows:

```bash
ssh root@CLOUD_STORAGE_IP
find /data/bert/ -type f | parallel -j4 --eta 'rsync -az --info=progress2 {} ubuntu@GPU_PROVIDER_IP:/home/ubuntu/data'
```

Replace `CLOUD_STORAGE_IP` and `GPU_PROVIDER_IP` with the IP addresses of your cloud storage and GPU provider machines, respectively.

That's it! You've now transferred your data to the GPU provider and are ready to start training your model.

