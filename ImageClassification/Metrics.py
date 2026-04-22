import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow import keras

def evaluate_model(model, x_test, y_test):
    # If labels are one-hot encoded, convert to integers
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    # Get predictions
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc, precision, recall, f1


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., None]

y_test = keras.utils.to_categorical(y_test, 10)

#######################################
#CHANGE HERE TO MODEL NAME FOR MEASURING DIFFERENT MODEL METRICS
model=keras.models.load_model("MNISTcnnv1.keras")
#######################################
evaluate_model(model, x_test, y_test)