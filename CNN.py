# Import necessary libraries
from keras.applications import *
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
import argparse

class DataGenerator(Sequence):
    def __init__(self, file_paths, quality_scores, batch_size):
        self.file_paths = file_paths
        self.quality_scores = quality_scores
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            if i < len(self.file_paths):
                img = image.load_img(self.file_paths[i], target_size=(224, 224))
                img = image.img_to_array(img)
                img = img / 255.0
                batch_x.append(img)
                batch_y.append(self.quality_scores[i])
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y


def create_model(model_name):
    if model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "ResNet101":
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "DenseNet121":
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "MobileNet":
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "MobileNetV3Large":
        base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "NASNetLarge":
        base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "NASNetMobile":
        base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "VGG19":
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Invalid model name")

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model


def train(model_name, model_architecture, labels_file_path, num_epochs, learning_rate, activation_function):
    model = create_model(model_architecture)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])

    df = pd.read_csv(labels_file_path)
    df.columns = ["Filename", 'QualityScore']

    train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)

    train_gen = DataGenerator(train_df['Filename'].values, train_df['QualityScore'].values, 32)
    test_gen = DataGenerator(validation_df['Filename'].values, validation_df['QualityScore'].values, 32)

    history = model.fit(
        train_gen,
        epochs=num_epochs,
        validation_data=test_gen,
        verbose=1
    )

    model.save(model_name + ".h5")


def test(model_file_path, labels_file_path):
    model = load_model(model_file_path)

    df = pd.read_csv(labels_file_path)
    df.columns = ["Filename"]

    test_gen = DataGenerator(df['Filename'].values, np.zeros(len(df)), 32)

    predictions = model.predict(test_gen)

    results = {
        'file_path': df['Filename'],
        'prediction': predictions.flatten()
    }

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"./{model_file_path[:-3]}_predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='Train or test a CNN model for regression')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Mode to run the script in')
    parser.add_argument('--model_name', type=str, help='Name of the model to use')
    parser.add_argument('--model_architecture', type=str, choices=[
        "ResNet50", "ResNet101", "InceptionResNetV2", "DenseNet121", "MobileNet", "MobileNetV2", "MobileNetV3Large",
        "NASNetLarge", "NASNetMobile", "VGG16", "VGG19"
    ], help='Architecture of the model to use')
    parser.add_argument('--labels_file_path', type=str, help='Path to the CSV file containing the dataset')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model for')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--activation_function', type=str, default='linear', help='Activation function for the output layer')
    parser.add_argument('--model_file_path', type=str, help='Path to the saved model file')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.model_name, args.model_architecture, args.labels_file_path, args.num_epochs, args.learning_rate, args.activation_function)
    elif args.mode == 'test':
        test(args.model_file_path, args.labels_file_path)


if __name__ == '__main__':
    main()
