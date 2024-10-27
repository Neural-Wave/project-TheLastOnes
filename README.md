# AI4Privacy | PII Detection and Masking with Language Models

This repository demonstrates a PII (Personally Identifiable Information) masking model based on transformers and datasets libraries. The model detects sensitive information like names, addresses, and phone numbers, and replaces them with designated labels for privacy protection.

## Dependencies

- Transformers: `pip install transformers`
- Datasets: `pip install datasets`
- Torch: `pip install torch`

## Running the Model

### 1. Inference on a single sentence: 

```python
# setup model and tokenizer correctly before you run this
example_text = "My name is Obee Nobi and I live at 432423 Deka St., Tanooti. My phone number is 5455 123 4567."

# tokenize and make predictions
inputs = tokenizer(example_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=-1)

# Use the `mask_pii` function from demo.ipynb for masking
masked_example_detailed = mask_pii(model, tokenizer, example_text, aggregate_redaction=False)
print(masked_example_detailed)
```


### 2.	Inference on a Dataset:

```bash
inference.py -s path-to-dataset
```

**Note:**  The script assumes that your dataset has a structure similar to the [PII Masking 400k dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-400k), which includes fields for raw text and label information.


## Highlight - Preprocessing

We did something what the [community asked for](https://huggingface.co/datasets/ai4privacy/pii-masking-400k/discussions/3):

- **`generate_sequence_labels(text, privacy_mask)`**:
	Creates source text-level labels based on PII entities in `privacy_mask`.

- **`tokenize_and_align_labels(examples)`**:
	Tokenizes text and align the token-level labels with the source text-level token

You can find them in `general_proprocess.ipynb`. 
Now you can play with the dataset with **ANY** kinds of models and tokenizers!

## Sturcture
- `Notebooks/`

  - `demo.ipynb`: the demo of running PPI masking on a text
 
  - `general_preprocess.ipynb`: the notebook that preprocess the data for general type of models an tokenizers
    
  - `preprocess.ipynb`: the notebook that preprocess data for bert model and tokenizer
    
- `training_script/`
  
  - `train.py`: finetuning the dataset on model "bert-base-multilingual-cased"
    
  - `train_gen.py`: finetuning the dataset on model "microsoft/mdeberta-v3-base"
    
  - `train_noise.py`: finetuning the dataset on model "microsoft/mdeberta-v3-base"
    
- `inference.py`: run inference on given dataset

## Performance

We reached the following performance of 3 models on the validation set of [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k)

<img width="513" alt="Screenshot 2024-10-27 at 12 48 22" src="https://github.com/user-attachments/assets/d9388921-d4e2-4301-ac74-d16408c0bc50">


## Future work

We also proposed a bert-like masking inputs strategy to hopefully improve the generalization ability of the model. However, we didn't have enough time to
futher fine tuning this model. Therefore, one possible future direction could be to explore the suitable hyperparameters to see if this strategy can avoid the situation of
overfitting.

## License
This project is licensed under the MIT License. See the LICENSE file for details.



