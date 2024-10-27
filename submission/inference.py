import re
import argparse
import torch
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
    Trainer, TrainingArguments
)

label_list = [
    'B-ACCOUNTNUM',
    'B-BUILDINGNUM',
    'B-CITY',
    'B-CREDITCARDNUMBER',
    'B-DATEOFBIRTH',
    'B-DRIVERLICENSENUM',
    'B-EMAIL',
    'B-GIVENNAME',
    'B-IDCARDNUM',
    'B-PASSWORD',
    'B-SOCIALNUM',
    'B-STREET',
    'B-SURNAME',
    'B-TAXNUM',
    'B-TELEPHONENUM',
    'B-USERNAME',
    'B-ZIPCODE',
    'I-ACCOUNTNUM',
    'I-BUILDINGNUM',
    'I-CITY',
    'I-CREDITCARDNUMBER',
    'I-DATEOFBIRTH',
    'I-DRIVERLICENSENUM',
    'I-EMAIL',
    'I-GIVENNAME',
    'I-IDCARDNUM',
    'I-PASSWORD',
    'I-SOCIALNUM',
    'I-STREET',
    'I-SURNAME',
    'I-TAXNUM',
    'I-TELEPHONENUM',
    'I-USERNAME',
    'I-ZIPCODE',
    'O',
]

id2label = {idx: label for idx, label in enumerate(label_list)}
label2id = {label: idx for idx, label in enumerate(label_list)}
label_set = set(l[2:] for l in label_list[:-1])

# choose our best model
MODEL_PATH = "./ckpt/modelv2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, truncation=True, max_length=512)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH, num_labels=35, id2label=id2label, label2id=label2id)

parser = argparse.ArgumentParser(description="Run inference on a specified dataset.")
parser.add_argument("-s", "--dataset", type=str, required=True, help="Path to the dataset directory")
args = parser.parse_args()
print("Dataset path:", args.dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

def generate_sequence_labels(text, privacy_mask):
    # sort privacy mask by start position
    privacy_mask = sorted(privacy_mask, key=lambda x: x['start'], reverse=True)
    
    # replace sensitive pieces of text with labels
    for item in privacy_mask:
        label = item['label']
        start = item['start']
        end = item['end']
        value = item['value']
        # count the number of words in the value
        word_count = len(value.split())
        
        # replace the sensitive information with the appropriate number of [label] placeholders
        replacement = " ".join([f"{label}" for _ in range(word_count)])
        text = text[:start] + replacement + text[end:]
        
    words = text.split()
    # assign labels to each word
    labels = []
    for word in words:
        match = re.search(r"(\w+)", word)  # match any word character
        if match:
            label = match.group(1)
            if label in label_set:
                labels.append(label)
            else:
                # any other word is labeled as "O"
                labels.append("O")
        else:
            labels.append("O")
    return labels


def tokenize_and_align_labels(examples):
    words = [t.split() for t in examples["source_text"]]
    tokenized_inputs = tokenizer(words, truncation=True, is_split_into_words=True, max_length=512)
    source_labels = [
        generate_sequence_labels(text, mask)
        for text, mask in zip(examples["source_text"], examples["privacy_mask"])
    ]

    labels = []
    valid_idx = []
    for i, label in enumerate(source_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # map tokens to their respective word.
        previous_label = None
        label_ids = [-100]
        try:
            for word_idx in word_ids:
                if word_idx is None:
                    continue
                elif label[word_idx] == "O":
                    label_ids.append(label2id["O"])
                    continue
                elif previous_label == label[word_idx]:
                    label_ids.append(label2id[f"I-{label[word_idx]}"])
                else:
                    label_ids.append(label2id[f"B-{label[word_idx]}"])
                previous_label = label[word_idx]
            label_ids = label_ids[:511] + [-100]
            labels.append(label_ids)
            # print(word_ids)
            # print(label_ids)
        except:
            # global k
            # k += 1
            # print(f"{word_idx = }")
            # print(f"{len(label) = }")
            labels.append([-100] * len(tokenized_inputs["input_ids"][i]))
        """
        except:
            print(f"{word_ids[-2] = }")
            print(f"{len(label) = }")
            print("Unvalid data detected")
            labels.append([-100] * len(word_ids))
        """

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# preprocess dataset
dataset = load_from_disk(args.dataset)
dataset = dataset.map(tokenize_and_align_labels, batched=True)
remove_columns = [
    'locale',
    'language',
    'split',
    'privacy_mask',
    'uid',
    'mbert_tokens',
    'mbert_token_classes',
    'source_text',
    'masked_text'
]
dataset = dataset.remove_columns(remove_columns)

seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


eval_args = TrainingArguments(
    output_dir="./eval",
    per_device_eval_batch_size=32,
    seed=42,
    bf16=True,
)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding="longest",
    max_length=512,
)

trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics    
)

evaluation_results = trainer.evaluate()

print(evaluation_results)