# evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

IMG_SIZE = (128,128)
BATCH_SIZE = 32
TEST_DIR = 'dataset/test'
MODEL_PATH = 'mask_detector.h5'
REPORT_CSV = 'classification_report.csv'
CM_CSV = 'confusion_matrix.csv'

def main():
    model = load_model(MODEL_PATH)
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    preds = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = (preds.ravel() > 0.5).astype(int)
    y_true = test_gen.classes[:len(y_pred)]
    class_indices = test_gen.class_indices
    # class_names ordered by index 0..n-1
    inv_map = {v:k for k,v in class_indices.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%\n")

    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(REPORT_CSV)
    print("Classification report saved to", REPORT_CSV)
    print(cr_df)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(CM_CSV)
    print("\nConfusion matrix saved to", CM_CSV)
    print(cm_df)

if __name__ == '__main__':
    main()
