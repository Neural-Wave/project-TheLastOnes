from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import DatasetDict
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments
import evaluate
import numpy as np

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

tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", truncation=True, max_length=512)
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/mdeberta-v3-base", num_labels=35, id2label=id2label, label2id=label2id)

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

privacy_dataset = load_from_disk("./tokenized_dataset/gen_tokenized_data")
remove_columns = [
    'locale',
    'split',
    'privacy_mask',
    'uid',
    'mbert_tokens',
    'mbert_token_classes',
    'source_text',
    'masked_text'
]
training_set = privacy_dataset["train"].remove_columns(remove_columns)
validation_set = privacy_dataset["validation"].remove_columns(remove_columns)

training_args = TrainingArguments(
    output_dir="./results-gen",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,
    seed=42
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest", max_length=512)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=validation_set,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()