import os
import datasets
from transformers import ViTFeatureExtractor
# from tensorflow import keras
# from tensorflow.keras import layers
import numpy as np
from huggingface_hub import HfFolder
import tensorflow as tf
from transformers import DefaultDataCollator
from transformers import TFViTForImageClassification, create_optimizer
from transformers import TFViTModel
from transformers.keras_callbacks import PushToHubCallback
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import torch
import json
from pathlib import Path


def get_config():
    with open("config.json") as f:
        config = json.load(f)

    return config

config = get_config()


def create_image_folder_dataset(root_path):
    """creates `Dataset` from image folder structure"""

    # get class names by folders names
    _CLASS_NAMES= os.listdir(root_path)
    # defines `datasets` features`
    features=datasets.Features({
                        "img": datasets.Image(),
                        "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
                    })
    # temp list holding datapoints for creation
    img_data_files=[]
    label_data_files=[]
    # load images into list for creation
    for img_class in os.listdir(root_path):
        for img in os.listdir(os.path.join(root_path,img_class)):
            path_=os.path.join(root_path,img_class,img)
            img_data_files.append(path_)
            label_data_files.append(img_class)
    # create dataset
    ds = datasets.Dataset.from_dict({"img":img_data_files,"label":label_data_files},features=features)
    return ds


def augmentation(examples):
    feature_extractor = ViTFeatureExtractor.from_pretrained(config["model_id"])

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(feature_extractor.size, feature_extractor.size),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    
    examples["pixel_values"] = [
        data_augmentation(np.array(image.convert("RGB"))) for image in examples["img"]
    ]

    return examples


def process(examples):
    feature_extractor = ViTFeatureExtractor.from_pretrained(config["model_id"])
    # Convert images to RGB if they aren't already
    rgb_images = [np.array(image.convert("RGB")) for image in examples["img"]]
    # Apply the feature extractor to the RGB images
    examples.update(feature_extractor(images=rgb_images, return_tensors="np"))
    return examples


def main():
    config = get_config()
    print(config)

if __name__ == "__main__":
    main()
