Our Implementation is based on `pytorch-22.09` *[1]*

## Setup Docker iamge & Data Preparation


### Step 1: Set up your environment

First, you'll need to have Docker and/or **NVIDIA-Docker[2]** installed on your machine. Docker is a platform that allows you to develop, ship, and run applications inside containers, and NVIDIA-Docker is a wrapper around Docker that provides a seamless integration with NVIDIA GPUs.


Clone the repository:

```bash
git clone --depth 1 https://github.com/mlcommons/training_results_v2.1.git
```


### Step 2: Build the Docker image

Navigate to the directory containing the Dockerfile:

```bash
cd ./training_results_v2.1/NVIDIA/benchmarks/bert/implementations/pytorch-22.09
```

Then, build the Docker image:

```bash
docker build -t mlperf-nvidia:language_model_v2.1_22.09 .
```

This command builds a Docker image using the Dockerfile in the current directory and tags it as `mlperf-nvidia:language_model_v2.1_22.09`.

### Step 3: Run the Docker container

Now, you can run a Docker container using the image you just built:

```bash
nvidia-docker run -it --shm-size=1g --ulimit memlock=-1 --privileged --ipc=host \
-v /data/bert/phase1:/workspace/phase1 \
-v /data/bert/hdf5/eval_varlength:/workspace/evaldata \
-v /ssd/hdf5_4320_shards_varlength:/workspace/data_phase2 \
-v /ssd/hdf5_4320_shards_varlength:/workspace/data \
-v /home/elim/bert_logs:/workspace/logs \
--name language_model_v2.1_22.09 mlperf-nvidia:language_model_v2.1_22.09
```

[1]: https://github.com/mlcommons/training_results_v2.1/tree/158189d4cbfbee366c10da1f0f086c85d8f15b5f/NVIDIA/benchmarks/bert/implementations/pytorch-22.09
[2]: https://github.com/NVIDIA/nvidia-docker/issues/1243
