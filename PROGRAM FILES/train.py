# train.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from model import build_cnn

# Config
IMG_SIZE = (128,128)
BATCH_SIZE = 32
EPOCHS = 15
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'
MODEL_PATH = 'mask_detector.h5'
HISTORY_CSV = 'history.csv'

def main():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    val_flow = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save final model (best weights already saved by checkpoint)
    model.save(MODEL_PATH)

    # Save history to CSV using pandas
    hist_df = pd.DataFrame(history.history)
    hist_df['epoch'] = hist_df.index + 1
    hist_df.to_csv(HISTORY_CSV, index=False)
    print(f"Training finished. Best model saved to: {MODEL_PATH}")
    print(f"Training history saved to: {HISTORY_CSV}")

    # Print best validation accuracy
    best_epoch = int(hist_df['val_accuracy'].idxmax()) + 1
    best_val_acc = hist_df['val_accuracy'].max()
    print(f"Best val_accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

if __name__ == '__main__':
    main()
