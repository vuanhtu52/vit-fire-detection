from preprocessing import *

import os
import datasets
from transformers import ViTFeatureExtractor
import numpy as np
from huggingface_hub import HfFolder
import tensorflow as tf
from transformers import DefaultDataCollator
from transformers import TFViTForImageClassification, create_optimizer
from transformers import TFViTModel
from transformers.keras_callbacks import PushToHubCallback
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import torch


class HuggingFaceCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, save_path):
        super().__init__()
        self.model = model
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        save_dir = f"{self.save_path}/epoch-{epoch + 1:02d}"
        print(f"\nSaving Hugging Face model to {save_dir}")
        self.model.save_pretrained(save_dir)


def train():
    config = get_config()

    ds_train = create_image_folder_dataset("data/train")
    img_class_labels = ds_train.features["label"].names

    # We are also renaming our label col to labels to use `.to_tf_dataset` later
    ds_train = ds_train.rename_column("label", "labels")

    ds_train_processed = ds_train.map(process, batched=True)

    test_size = 0.2
    ds_train_processed = ds_train_processed.shuffle().train_test_split(test_size=test_size)

    id2label = {str(i): label for i, label in enumerate(img_class_labels)}
    label2id = {v: k for k, v in id2label.items()}
    
    num_train_epochs = config["epochs"]
    train_batch_size = config["batch_size"]
    eval_batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    weight_decay_rate= config["weight_decay_rate"]
    num_warmup_steps= config["num_warmup_steps"]
    output_dir=config["model_id"].split("/")[1] 
    hub_token = HfFolder.get_token() # or your token directly "hf_xxx"
    hub_model_id = f'{config["model_id"].split("/")[1]}-fire'
    fp16=True

    if fp16:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Data collator that will dynamically pad the inputs received, as well as the labels.
    data_collator = DefaultDataCollator(return_tensors="tf")
    
    # converting our train dataset to tf.data.Dataset
    tf_train_dataset = ds_train_processed["train"].to_tf_dataset(
    columns=['pixel_values'],
    label_cols=["labels"],
    shuffle=True,
    batch_size=train_batch_size,
    collate_fn=data_collator)
    
    # converting our test dataset to tf.data.Dataset
    tf_eval_dataset = ds_train_processed["test"].to_tf_dataset(
    columns=['pixel_values'],
    label_cols=["labels"],
    shuffle=True,
    batch_size=eval_batch_size,
    collate_fn=data_collator)

    # create optimizer with weigh decay
    num_train_steps = len(tf_train_dataset) * num_train_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=weight_decay_rate,
        num_warmup_steps=num_warmup_steps,
    )
    
    # load pre-trained ViT model
    model = TFViTForImageClassification.from_pretrained(
        config["model_id"],
        num_labels=len(img_class_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # define loss
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # define metrics
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
    ]
    
    # compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # # Define the checkpoint callback
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f"{config['output_dir']}/checkpoints/epoch-{{epoch:02d}}.ckpt",  # Save the model with the epoch number in the filename
    #     save_weights_only=False,                         
    #     save_freq='epoch',                               # Save after every epoch
    #     verbose=1                                        # Print a message when saving the checkpoint
    # )
    # Define the Hugging Face checkpoint callback
    hf_checkpoint_callback = HuggingFaceCheckpointCallback(
        model, config['output_dir'] + '/checkpoints'
    )

    train_results = model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset,
        epochs=num_train_epochs,
        callbacks=[hf_checkpoint_callback]
    )


def main():
    train()

if __name__ == "__main__":
    main()