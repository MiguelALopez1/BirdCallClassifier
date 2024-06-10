from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from datasets import Dataset, ClassLabel
import numpy as np
import evaluate

def train_model(df):
    TEST_SIZE = 0.1
    labels_list = sorted(list(df['label'].unique()))

    label2id, id2label = {}, {}
    for i, label in enumerate(labels_list):
        label2id[label] = i
        id2label[i] = label

    ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

    def map_label2id(example):
        example['label'] = ClassLabels.str2int(example['label'])
        return example

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(map_label2id, batched=True)
    dataset = dataset.cast_column('label', ClassLabels)
    dataset = dataset.train_test_split(test_size=TEST_SIZE, shuffle=True, stratify_by_column="label")

    model_str = "facebook/wav2vec2-base-960h"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_str)
    model = AutoModelForAudioClassification.from_pretrained(model_str, num_labels=len(labels_list))
    model.config.id2label = id2label

    def preprocess_function(batch):
        inputs = feature_extractor(batch['audio'], sampling_rate=16000, max_length=16000*10, truncation=True)
        inputs['input_values'] = inputs['input_values'][0]
        return inputs

    dataset['train'] = dataset['train'].map(preprocess_function, remove_columns="audio", batched=False)
    dataset['test'] = dataset['test'].map(preprocess_function, remove_columns="audio", batched=False)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        predictions = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
        label_ids = eval_pred.label_ids
        acc_score = accuracy.compute(predictions=predictions.argmax(axis=1), references=label_ids)['accuracy']
        return {"accuracy": acc_score}

    training_args = TrainingArguments(
        output_dir="bird_sounds_classification",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-6,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    return trainer, dataset['test']
