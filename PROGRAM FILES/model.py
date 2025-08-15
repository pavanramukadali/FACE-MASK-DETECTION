# model.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def build_cnn(input_shape=(128,128,3)):
    # Load MobileNetV2 without the top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))  # binary classification

    return model
