Our Implementation is based on `pytorch-22.09` *[1]*

After setting up the image and the data, we can run the pretraining now :)

### Step 1: Environment variables

Update `config_A30xxx.sh`, remove a few flags, and rename it `config_SUT.sh`

```
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0
export CUDA_VISIBLE_DEVICES=0,1
export NEXP=10
source config_SUT.sh
```


### Step 5: Run the BERT training script

```
./run_and_time.sh
```

??? expected_log
    ```
    :::MLLOG {"namespace": "", "time_ms": 1684505482468, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.7091760635375977, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 1210432}}
    {'global_steps': 2679, 'eval_loss': 1.3657647371292114, 'eval_mlm_accuracy': 0.7091760635375977}
    :::MLLOG {"namespace": "", "time_ms": 1684506469845, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.711473822593689, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 1361344}}
    {'global_steps': 3014, 'eval_loss': 1.3529666662216187, 'eval_mlm_accuracy': 0.711473822593689}
    :::MLLOG {"namespace": "", "time_ms": 1684507306048, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.7122747898101807, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 1512256}}
    {'global_steps': 3349, 'eval_loss': 1.3459392786026, 'eval_mlm_accuracy': 0.7122747898101807}
    :::MLLOG {"namespace": "", "time_ms": 1684508147669, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.7132415771484375, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 1663168}}
    {'global_steps': 3684, 'eval_loss': 1.342519760131836, 'eval_mlm_accuracy': 0.7132415771484375}
    :::MLLOG {"namespace": "", "time_ms": 1684509073482, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.714259684085846, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 1814048}}
    {'global_steps': 4018, 'eval_loss': 1.3379616737365723, 'eval_mlm_accuracy': 0.714259684085846}
    :::MLLOG {"namespace": "", "time_ms": 1684509946280, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.7151680588722229, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 1964960}}
    {'global_steps': 4353, 'eval_loss': 1.3296408653259277, 'eval_mlm_accuracy': 0.7151680588722229}
    :::MLLOG {"namespace": "", "time_ms": 1684510743823, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.715915322303772, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 2115872}}
    {'global_steps': 4688, 'eval_loss': 1.327706217765808, 'eval_mlm_accuracy': 0.715915322303772}
    :::MLLOG {"namespace": "", "time_ms": 1684511646616, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.7172977328300476, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 2266784}}
    {'global_steps': 5023, 'eval_loss': 1.3210091590881348, 'eval_mlm_accuracy': 0.7172977328300476}
    :::MLLOG {"namespace": "", "time_ms": 1684512554140, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.7179118990898132, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1722, "epoch_num": 2417696}}
    {'global_steps': 5358, 'eval_loss': 1.3161648511886597, 'eval_mlm_accuracy': 0.7179118990898132}

    ```
