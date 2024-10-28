import tensorflow as tf
import json
import os
import numpy as np
from transformers import DefaultDataCollator, ViTFeatureExtractor, TFViTForImageClassification
import datasets
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def get_config():
    with open("config.json") as f:
        config = json.load(f)

    return config

config = get_config()


def create_image_folder_dataset(root_path):
    """Creates a `Dataset` from an image folder structure."""
    # Get class names from folder names
    _CLASS_NAMES = os.listdir(root_path)
    # Define `datasets` features
    features = datasets.Features({
        "img": datasets.Image(),
        "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
    })
    # Temp lists holding data points for creation
    img_data_files = []
    label_data_files = []
    # Load images into the list for creation
    for img_class in os.listdir(root_path):
        for img in os.listdir(os.path.join(root_path, img_class)):
            path_ = os.path.join(root_path, img_class, img)
            img_data_files.append(path_)
            label_data_files.append(img_class)
    # Create dataset
    ds = datasets.Dataset.from_dict({"img": img_data_files, "label": label_data_files}, features=features)
    return ds


def process(examples):
    feature_extractor = ViTFeatureExtractor.from_pretrained(config["model_id"])
    # Convert images to RGB if they aren't already
    rgb_images = [np.array(image.convert("RGB")) for image in examples["img"]]
    # Apply the feature extractor to the RGB images
    examples.update(feature_extractor(images=rgb_images, return_tensors="np"))
    return examples


def load_model_checkpoint(checkpoint_path, num_labels, id2label, label2id):
    # Load the model from the specified checkpoint
    model = TFViTForImageClassification.from_pretrained(
        checkpoint_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def generate_classification_report(model, tf_eval_dataset, class_names):
    # Get the ground truth labels
    y_true = []
    for batch in tf_eval_dataset:
        _, labels = batch
        y_true.extend(labels.numpy())

    # Get the predicted labels
    y_pred = []
    for batch in tqdm(tf_eval_dataset):
        images, _ = batch
        predictions = model.predict(images)  # Get the raw prediction scores (logits)
        logits = predictions.logits  # Extract the logits
        y_pred_batch = tf.argmax(logits, axis=1).numpy()  # Convert logits to predicted class indices
        y_pred.extend(y_pred_batch)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)


def plot_confusion_matrix(cm, class_names):
    """Plots the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate_model():
    config = get_config()

    # Load the dataset
    ds_test = create_image_folder_dataset("data/test")
    img_class_labels = ds_test.features["label"].names

    # Rename the label column to labels
    ds_test = ds_test.rename_column("label", "labels")

    # Process the test dataset
    ds_test_processed = ds_test.map(process, batched=True)

    # Prepare id2label and label2id mappings
    id2label = {str(i): label for i, label in enumerate(img_class_labels)}
    label2id = {v: k for k, v in id2label.items()}

    # Create the test dataset for TensorFlow
    eval_batch_size = 32
    data_collator = DefaultDataCollator(return_tensors="tf")
    tf_eval_dataset = ds_test_processed.to_tf_dataset(
        columns=['pixel_values'],
        label_cols=["labels"],
        shuffle=False,
        batch_size=eval_batch_size,
        collate_fn=data_collator
    )

    # Load the model checkpoint
    # checkpoint_path = f"{config['output_dir']}/checkpoints/epoch-01"  
    # model = load_model_checkpoint(
    #     checkpoint_path,
    #     num_labels=len(img_class_labels),
    #     id2label=id2label,
    #     label2id=label2id
    # )
    # Load the model from the checkpoint
    checkpoint_path = f"{config['output_dir']}/checkpoints/epoch-01"  # Adjust based on where your model is saved
    model = TFViTForImageClassification.from_pretrained(
        checkpoint_path,
        num_labels=len(img_class_labels),
        id2label={str(i): label for i, label in enumerate(img_class_labels)},
        label2id={label: str(i) for i, label in enumerate(img_class_labels)},
    )

    # Compile the model with the same loss and metrics as used during training
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # metrics = [
    #     tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    #     tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
    # ]

    # model.compile(
    #     loss=loss,
    #     metrics=metrics
    # )
    # Compile the model
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss)

    # Evaluate the model on the test set
    # print("\nEvaluating the model on the test set:")
    # evaluation_results = model.evaluate(tf_eval_dataset)
    # print(f"Test loss: {evaluation_results[0]}")
    # print(f"Test accuracy: {evaluation_results[1]}")
    # return evaluation_results

    # Evaluate the model and generate a classification report
    generate_classification_report(model, tf_eval_dataset, img_class_labels)


if __name__ == "__main__":
    evaluate_model()
