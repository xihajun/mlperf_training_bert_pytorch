# Training a BERT Model with PyTorch


This document provides a step-by-step guide to fine-tuning a BERT model using PyTorch, complete with code for data loading, model training, and evaluation. This tutorial also covers the use of multiple GPUs for training.


[![Open In Colab](colab-badge.svg)](https://colab.research.google.com/drive/1PqzIhHt6K6j5HQ8uYA5qcAjW9Vf-6hQy?usp=sharing]


## Step 1: Import necessary libraries
Firstly, import the necessary libraries and modules.

```python
import os
import torch
import h5py
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import BertForPreTraining, AdamW, get_linear_schedule_with_warmup
```

## Step 2: Define the dataset
Create a custom dataset class for loading data from the HDF5 files. This will be used to load our preprocessed data.

The provided code handles the loading of the input files and their respective labels. It also pads the sequences to the max sequence length.

```python
class HDF5DatasetForNextSentencePrediction(Dataset):
    # ... (refer to your provided code) ...
```
??? Expand
    ```
    class HDF5DatasetForNextSentencePrediction(Dataset):
        def __init__(self, data_dir, tokenizer, max_seq_length):
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length

            input_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.hdf5')]
            input_files.sort()

            self.input_ids = []
            self.segment_ids = []
            self.attention_mask = []
            self.next_sentence_labels = []
            self.masked_lm_positions = []
            self.masked_lm_ids = []

            for input_file in input_files:
                with h5py.File(input_file, 'r') as f:
                    self.input_ids.extend(f['input_ids'])
                    self.segment_ids.extend(f['segment_ids'])
                    self.attention_mask = [np.ones_like(ids) for ids in self.input_ids]  # Assuming all tokens have attention_mask = 1
                    self.next_sentence_labels.extend(f['next_sentence_labels'])
                    self.masked_lm_positions.extend(f['masked_lm_positions'])
                    self.masked_lm_ids.extend(f['masked_lm_ids'])

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            def pad_to_max_length(arr, max_length):
                pad_len = max_length - len(arr)
                return np.pad(arr, (0, pad_len), mode='constant') if pad_len > 0 else arr[:max_length]

            input_ids = pad_to_max_length(self.input_ids[idx], self.max_seq_length)
            token_type_ids = pad_to_max_length(self.segment_ids[idx], self.max_seq_length)
            attention_mask = pad_to_max_length(self.attention_mask[idx], self.max_seq_length)
            masked_lm_positions = pad_to_max_length(self.masked_lm_positions[idx], self.max_seq_length)
            masked_lm_ids = pad_to_max_length(self.masked_lm_ids[idx], self.max_seq_length)

            return {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'next_sentence_label': torch.tensor(self.next_sentence_labels[idx], dtype=torch.long),
                    'masked_lm_positions': torch.tensor(masked_lm_positions, dtype=torch.long),
                    'masked_lm_ids': torch.tensor(masked_lm_ids, dtype=torch.long),
                }
    ```


## Step 3: Initialize the dataset and data loader
Initialize the dataset and data loader using the custom dataset class. 

```python
MAX_SEQ_LENGTH = 512
DATA_DIR = '/path/to/hdf5/files' 
BATCH_SIZE = 16 

train_dataset = HDF5DatasetForNextSentencePrediction(
    data_dir=DATA_DIR,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
```

## Step 4: Initialize the model, optimizer, and scheduler
Next, we'll initialize the model, the optimizer, and the learning rate scheduler.

```python
model = BertForPreTraining.from_pretrained('bert-base-uncased')

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
)

total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
```

## Step 5: Set up multiple GPUs
If multiple GPUs are available, we can use PyTorch's `nn.DataParallel` to parallelize the model.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)
```

## Step 6: Train the model
Now we can start training the model. In the training loop, we'll calculate both the masked language model loss and the next sentence prediction loss. 

We'll also evaluate the model after each training epoch using the provided evaluation function.

```python
for epoch in range(EPOCHS):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # ... Training code here ...

    # Save the model after each epoch
    model.module.save_pretrained(f'trained_bert_epoch_{epoch + 1}') 

    # Evaluation after each epoch
    print("Start evaluating...")
    mlm_evaluation(model.module, eval_dataset, tokenizer)
```

??? Expand
   ```
   for step, batch in enumerate(train_dataloader):

        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        masked_lm_positions = batch['masked_lm_positions'].to(device)
        masked_lm_ids = batch['masked_lm_ids'].to(device)
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        prediction_scores, seq_relationship_score = outputs.prediction_logits, outputs.seq_relationship_logits

        masked_lm_loss_fct = CrossEntropyLoss(ignore_index=-1)  # Ignore padding tokens
        masked_lm_labels = torch.full(masked_lm_positions.size(), fill_value=-1, dtype=torch.long, device=device)
        masked_lm_labels.scatter_(1, masked_lm_positions, masked_lm_ids)

        masked_lm_loss = masked_lm_loss_fct(prediction_scores.view(-1, model.config.vocab_size), masked_lm_labels.view(-1))

        next_sentence_loss_fct = CrossEntropyLoss()
        next_sentence_loss = next_sentence_loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))

        total_loss = masked_lm_loss + next_sentence_loss

        # Perform a backward pass to calculate the gradients
        total_loss.backward()

        # Update the weights
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        # Clear the gradients
        optimizer.zero_grad()

        # Print the loss for every `print_interval` steps
        if step % print_interval == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Step {step}, Loss: {total_loss.item()}")
   ```


## Step 7: Define the evaluation function
This function computes the MLM loss and calculates the accuracy of the model after each epoch.

```python
def mlm_evaluation(model, dataset, tokenizer, batch_size=16):
    # ... (refer to the provided code) ...
```

??? Expand
    ```

    def mlm_evaluation(model, dataset, tokenizer, batch_size=16):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        model.eval()

        total_loss = 0.0
        total_corrects = 0
        total_words = 0
        count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                masked_lm_positions = batch['masked_lm_positions'].to(device)
                masked_lm_ids = batch['masked_lm_ids'].to(device)

                outputs = model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
                prediction_scores, _ = outputs.prediction_logits, outputs.seq_relationship_logits

                masked_lm_loss_fct = CrossEntropyLoss(ignore_index=-1)  # Ignore padding tokens
                masked_lm_labels = torch.full(masked_lm_positions.size(), fill_value=-1, dtype=torch.long, device=device)
                masked_lm_labels.scatter_(1, masked_lm_positions, masked_lm_ids)

                masked_lm_loss = masked_lm_loss_fct(prediction_scores.view(-1, model.config.vocab_size), masked_lm_labels.view(-1))

                total_loss += masked_lm_loss.item()

                # Calculating accuracy
                predictions = torch.argmax(prediction_scores, dim=2)
                total_corrects += ((predictions == masked_lm_labels) & (masked_lm_labels != -1)).sum().item()
                total_words += (masked_lm_labels != -1).sum().item()


        print(f"Loss: {total_loss / len(dataloader)}")
        print(f"Accuracy: {total_corrects / total_words}")
        return total_loss / len(dataloader), total_corrects / total_words

    ```
Now you have a complete script for training a BERT model on your custom data. You can load your trained model using `mlm_evaluation`.

## Step 8: Evaluate the model
Let's evaluate the model on the validation set.

```python
eval_dataset = HDF5DatasetForNextSentencePrediction(
    data_dir=EVAL_DATA_DIR,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
)

loss, acc = mlm_evaluation(model, eval_dataset, tokenizer, batch_size=32)
```
