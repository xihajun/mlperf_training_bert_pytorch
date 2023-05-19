
The goal is the preprocess the data into smaller chunks, e.g. 270, instead of 4320 so that it can be loaded onto the GPU much faster.

Assuming on chai.

### Setup
```
mkdir -p /data/bert-debug/hdf5/training

scp /data/bert/hdf5/training/part-0000*-of-00500.hdf5 .

scp /data/bert/download/result4/part-0000*-of-00500.hdf5 .
```

```
git clone --depth 1 git@github.com:mlcommons/training_results_v2.1.git
cd ~/training_results_v2.1/NVIDIA/benchmarks/bert/implementations/pytorch-22.09
docker build -t mlperf-nvidia:language_model_v2.1_22.09 .

docker run -it -v /data/bert-debug-temp:/workspace/bert_data --name bert_data mlperf-nvidia:language_model_v2.1_22.09
```

### Step 1
```
./input_preprocessing/parallel_create_hdf5.sh -i /workspace/bert_data/download/results4 -o /workspace/bert_data/hdf5/training -v /workspace/bert_data/phase1/vocab.txt
```

### Step 2
```
export SHARDS=270

python3 /workspace/bert/input_preprocessing/chop_hdf5_files.py \
 --num_shards ${SHARDS} \
 --input_hdf5_dir /workspace/bert_data/hdf5/training \
 --output_hdf5_dir /workspace/bert_data/hdf5/training-${SHARDS}
```

Output:
```
root@ff1c30161653:/workspace/bert/input_preprocessing# python3 /workspace/bert/input_preprocessing/chop_hdf5_files.py \
>  --num_shards ${SHARDS} \
_hdf5_>  --input_hdf5_dir /workspace/bert_data/hdf5/training \
>  --output_hdf5_dir /workspace/bert_data/hdf5/training-${SHARDS}
INFO:root:n_input_shards = 10
INFO:root:n_output_shards = 270
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 530.37it/s]
INFO:root:Total number of samples: 1925732. Sample per shard 7132/7133
INFO:root:creating 270 output file handles.  This could take a while.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:22<00:00,  2.28s/it
```

### Step 3
Shuffle
```
mkdir -p /workspace/bert_data/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_varlength

python3 /workspace/bert/input_preprocessing/shuffle_samples.py \
--input_hdf5 /workspace/bert_data/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_uncompressed \
--output_hdf5 /workspace/bert_data/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_varlength
```

```
256/270 [00:12<00:00, 26.94it/s]<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
 96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎     | 259/270 [00:12<00:00, 26.39it/s]<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊    | 262/270 [00:13<00:00, 25.64it/s]<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
 98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍  | 265/270 [00:13<00:00, 23.60it/s]<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
 99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉ | 268/270 [00:13<00:00, 24.03it/s]<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
<KeysViewHDF5 ['input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'next_sentence_labels', 'segment_ids']>
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 270/270 [00:13<00:00, 20.08it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 270/270 [00:00<00:00, 348.22it/s
```

### Step 4
write shuffled samples
```
python3 /workspace/bert/input_preprocessing/shuffle_samples_write.py \
  --input_hdf5 /workspace/bert_data/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_uncompressed \
  --output_hdf5 /workspace/bert_data/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_varlength
```
wait

### Step 5
Base point for eval
```
elim@chai:/data/bert-debug/hdf5/eval$ scp /data/bert/hdf5/eval/eval_all.hdf5 .
```

Pick 500 samples for eval
```
python3 /workspace/bert/input_preprocessing/pick_eval_samples.py \
 --input_hdf5_file=/workspace/bert_data/hdf5/eval/eval_all.hdf5 \
 --output_hdf5_file=/workspace/bert_data/hdf5/eval/part_eval_10k \
 --num_examples_to_pick=500
```

### Step 6
```
mkdir -p /workspace/bert_data/hdf5/eval_varlength
python3 /workspace/bert/input_preprocessing/convert_fixed2variable.py --input_hdf5_file /workspace/bert_data/hdf5/eval/part_eval_10k.hdf5 \
--output_hdf5_file /workspace/bert_data/hdf5/eval_varlength/part_eval_10k.hdf5
```

```
root@7d77a73dcf17:/workspace/bert# python3 /workspace/bert/input_preprocessing/convert_fixed2variable.py --input_hdf5_file /workspace/bert_data/hdf5/eval/part_eval_10k.hdf5 \
> --output_hdf5_file /workspace/bert_data/hdf5/eval_varlength/part_eval_10k.hdf5
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 446.79it/s]
INFO:root:Converted 500 examples in 1.1 sec
```

### Step 7
outside container. download model
```
elim@chai$ 

mkdir /data/bert-debug/phase1/
# bert_config.json
gdown https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW
# model.ckpt-28252.data-00000-of-00001
gdown https://drive.google.com/uc?id=1chiTBljF0Eh1U5pKs6ureVHgSbtU8OG_
# model.ckpt-28252.index
gdown https://drive.google.com/uc?id=1Q47V3K3jFRkbJ2zGCrKkKk-n0fvMZsa0
# model.ckpt-28252.meta
gdown https://drive.google.com/uc?id=1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv

scp /data/bert/phase1/model.ckpt-28252.data-00000-of-00001 /data/bert-debug/phase1/ &&\
scp /data/bert/phase1/model.ckpt-28252.index /data/bert-debug/phase1/ &&\
scp /data/bert/phase1/model.ckpt-28252.meta /data/bert-debug/phase1/ &&\
scp /data/bert/phase1/bert_config.json /data/bert-debug/phase1/
```

### Step 8
inside container. convert model
```
python3 /workspace/bert/convert_tf_checkpoint.py \
  --tf_checkpoint /workspace/bert_data/phase1/model.ckpt-28252 \
  --bert_config_path /workspace/bert_data/phase1/bert_config.json \
  --output_checkpoint /workspace/bert_data/phase1/model.ckpt-28252.pt
```