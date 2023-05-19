

### Issue: Multi-GPU Not Working

The multi-GPU setup was not working correctly.

#### Solution:

This issue was due to the incorrect configuration of the PyTorch distributed launch. To resolve this issue, we need to add the `-m torch.distributed.launch --nproc_per_node=2 --use_env` flags to the command that launches the script.

The corrected command should look like this:

```bash
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --log_dir=log /workspace/bert/run_pretraining.py --train_batch_size=112 --learning_rate=3.7e-2 --opt_lamb_beta_1=0.9 --opt_lamb_beta_2=0.999 --warmup_proportion=0.0 --warmup_steps=0.0 --start_warmup_step=0 --max_steps=7100 --phase2 --max_seq_length=512 --max_predictions_per_seq=76 --input_dir=/workspace/data_phase2 --init_checkpoint=/workspace/phase1/model.ckpt-28252.pt --do_train --skip_checkpoint --train_mlm_accuracy_window_size=0 --target_mlm_accuracy=0.720 --weight_decay_rate=0.01 --max_samples_termination=20000000 --eval_iter_start_samples=150000 --eval_iter_samples=150000 --eval_batch_size=16 --eval_dir=/workspace/evaldata --num_eval_examples 10000 --cache_eval_data --output_dir=/results --fp16 --distributed_lamb --dwu-num-rs-pg=1 --dwu-num-ar-pg=1 --dwu-num-ag-pg=1 --dwu-num-blocks=1 --gradient_accumulation_steps=14 --log_freq=1000 --bert_config_path=/workspace/phase1/bert_config.json --allreduce_post_accumulation --allreduce_post_accumulation_fp16
```

The `--nproc_per_node=2` flag specifies the number of GPUs to use on each node, and the `--use_env` flag tells the script to use the environment variables for configuration.

Remember to always check your command line arguments and ensure they are correctly configured for your specific setup. 

Change `run_and_time.sh` on line 110: https://github.com/mlcommons/training_results_v2.1/blob/158189d4cbfbee366c10da1f0f086c85d8f15b5f/NVIDIA/benchmarks/bert/implementations/pytorch-22.09/run_and_time.sh#L110

### Issue: Slow Training on A5000

We observed that the training on A5000 using our script was slower than expected. 

#### Solution:

We explored different GPU cloud providers and found that Lambda Labs provided the best performance for our needs. We considered the H100 and 1xA100 options due to their cost-effectiveness and performance. 

### Issue: Model Not Converging

Despite setting up everything on the H100, including data transfer and image building, the model did not converge during training. 

#### Possible Causes:

1. **Issues with the Converted Model:** We converted the TensorFlow model to PyTorch using NVIDIA’s script. However, we encountered errors indicating that the initial weights were not loaded properly. This could be due to differences in the naming systems used by TensorFlow and PyTorch.

2. **High MLM Loss:** The loss for BERT includes MLM loss and NSP loss. The MLM loss could be quite high, which might affect model convergence.

#### Solutions:

1. **Investigate the Converted Model:** We plan to investigate potential issues with the TensorFlow model conversion. This includes checking whether the conversion was done properly and whether the initial weights were loaded correctly.

2. **Adjust Training Data:** We plan to use fewer but larger files for training. This could help improve the MLM and potentially lead to increasing numbers.

3. **Longer Training Time:** We plan to train the model for a longer period and evaluate its MLM accuracy.

### Issue: Low Storage Space

We encountered low storage space on the chai server, which could affect the performance of the training.

#### Solution:

We cleaned the cache and deleted unnecessary images to free up storage space. We also pruned the Docker system to remove all stopped containers, unused networks, and images without associated containers.

### Issue: CUDA Out of Memory

We encountered a CUDA out of memory error during training. This could be due to the GPU memory being insufficient for the training process.

#### Solution:

We may have added some CUDA sync/memory thing in the old container, which could handle more data but might also be the reason why the model is not training. We need to investigate this further and consider solutions such as reducing the batch size or using a GPU with more memory.

### Issue: Long Training Time

The training process seems to require a large number of epochs, which could lead to a long training time.

#### Solution:

We need to investigate whether the number of epochs is appropriate for our training process. If necessary, we could consider adjusting the number of epochs or using techniques such as early stopping to reduce the training time.


### Issue: Different Loss under v2.1 Settings

We observed a difference in loss under the v2.1 settings, even though we were training the same dataset repeatedly.

#### Solution:

We need to investigate why the loss is different under the v2.1 settings. This could be due to differences in the training process or the model parameters.

### Issue: Complicated Code with Distributed GPU Optimization

The code for the training process is quite complicated, with a lot of optimization for distributed GPUs and shared memories. This could make it difficult to understand and troubleshoot the training process.

#### Solution:

We need to understand the code and the optimization techniques used. This could help us identify potential issues and improve the training process.

### Issue: Data Loading Process

The data loading process seemed to be a bit weird. It only loads the first dataset in the file system and does so repeatedly. This could be an issue, as it would mean the model is only being trained on a small subset of the data.

#### Solution:

We added a new line in the code to ensure that different data is loaded during the training process. This could help improve the model's performance by exposing it to a wider range of data.

### Issue: High Loss

The average loss was dropping initially but then increased to 4.59. The evaluation loss was not changing at all.

#### Possible Causes:

1. **Issues with Data Loading:** The model might be trained on the same small subset of data repeatedly, which could lead to overfitting and high loss.

2. **Issues with Evaluation Data:** There might be issues with the evaluation data, which could affect the evaluation loss.

#### Solutions:

1. **Improve Data Loading Process:** We need to ensure that the model is exposed to new data continuously during training. This could help prevent overfitting and reduce the loss.

2. **Check Evaluation Data:** We need to check the evaluation data and ensure that it is appropriate for evaluating the model's performance.

### Issue: Small Files for Training

NVIDIA separates files into many small files for training. This could be due to the large number of GPUs they have to process the data.

#### Solution:

We need to understand why NVIDIA separates files into many small files for training. This could help us optimize our own training process and make better use of our resources.


### Issue: Docker Image Not Working Well

The docker image was not working well, and an error was thrown related to the Apex library and the compute capability of the GPU.

??? Error
    ```
        Traceback (most recent call last):
    File "/workspace/bert/run_pretraining.py", line 1880, in <module>
        args, final_loss, train_time_raw = main()
    File "/workspace/bert/run_pretraining.py", line 1388, in main
        loss, mlm_acc, _ = fwd_loss_bwd_trainer.step(-1,
    File "/workspace/bert/fwd_loss_bwd_trainer.py", line 154, in step
        loss, mlm_acc, _ = model(*batch)
    File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in _call_impl
        return forward_call(*input, **kwargs)
    File "/workspace/bert/modeling.py", line 1181, in forward
        sequence_output, pooled_output = self.bert_model_segment(input_ids, token_type_ids, attention_mask)
    File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in _call_impl
        return forward_call(*input, **kwargs)
    File "/workspace/bert/modeling.py", line 1121, in forward
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, position_ids,
    File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in _call_impl
        return forward_call(*input, **kwargs)
    File "/workspace/bert/modeling.py", line 1014, in forward
        encoded_layers = self.encoder(embedding_output,
    File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in _call_impl
        return forward_call(*input, **kwargs)
    File "/workspace/bert/modeling.py", line 690, in forward
        hidden_states = layer_module(hidden_states, cu_seqlens, maxseqlen_in_batch)
    File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in _call_impl
        return forward_call(*input, **kwargs)
    File "/workspace/bert/modeling.py", line 621, in forward
        attention_output = self.attention(hidden_states, attention_mask, seqlen, batch)
    File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in _call_impl
        return forward_call(*input, **kwargs)
    File "/workspace/bert/modeling.py", line 493, in forward
        self_output = self.self(input_tensor, cu_seqlens, max_s, is_training=self.training)
    File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in _call_impl
        return forward_call(*input, **kwargs)
    File "/workspace/bert/fmha.py", line 214, in forward
        ctx = FMHAFun.apply(qkv.view(-1, 3, self.h, self.d), cu_seqlens, p_dropout, max_s, is_training, self.set_zero)
    File "/workspace/bert/fmha.py", line 34, in forward
        context, S_dmask = mha.fwd(qkv, cu_seqlens, p_dropout, max_s, is_training, is_nl, set_zero, None)
    RuntimeError: Expected (dprops->major == 8 && dprops->minor == 0) || (dprops->major == 9 && dprops->minor == 0) to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
    ```
#### Solution:

This issue was due to certain flags being enabled in the [`config_A30xx.sh` file](https://github.com/mlcommons/training_results_v2.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09/config_A30_1x2x224x14.sh). To resolve this issue, we need to remove these flags from the file. The flags that need to be removed are --unpad_fmha, --fused_bias_fc, --fused_bias_mha, and --fused_dropout_add.

