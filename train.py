
import time
import numpy as np
from data_loader import load_labels, load_series, normalize, make_sequences
from lstm import LSTM

# Configurations/Hyperparameters
DATA_PATH = "data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv"
FILE_KEY = "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv"
LABELS_PATH = "data/labels/combined_windows.json"

WIN_SIZE = 20
HIDDEN_DIM = 16
LR = 0.01
EPOCHS = 5



def main():
    #data pipeline 
    labels_dict = load_labels(LABELS_PATH)
    df = load_series(DATA_PATH, FILE_KEY, labels_dict)
    print(f"Loaded dataset: {len(df)} rows ({df['label'].sum()} anomalies)") #how many actual anomaly points exist

    values = df["value"].values.astype(float)
    norm_vals, _, _ = normalize(values)

    X, y, idx = make_sequences(norm_vals, WIN_SIZE)
    labels = df["label"].values[idx]
#Dataset splitting 
    split = int(len(X) * 0.6)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    test_labels = labels[split:]
#training loop
    model = LSTM(input_size=1, hidden_size=HIDDEN_DIM, output_size=1)

    print("Training model...")
    t0 = time.time() #epoch start time
    for ep in range(EPOCHS): #5 times loop
        ep_losses = []
        for i in range(len(X_train)):
            loss = model.train_step(X_train[i], y_train[i], LR)
            ep_losses.append(loss)
        print(f"Epoch {ep+1}/{EPOCHS} - loss: {np.mean(ep_losses):.4f}")
    
    print(f"Training completed in {time.time() - t0:.2f}s")

    # Predict on test set
    preds = np.array([model.predict(x)[0] for x in X_test])
    errors = (preds - y_test.flatten()) ** 2

    thresh = np.percentile(errors, 95)
    anomalies = (errors > thresh).astype(int)

    # Calculate metrics
    tp = np.sum((anomalies == 1) & (test_labels == 1))
    fp = np.sum((anomalies == 1) & (test_labels == 0))
    fn = np.sum((anomalies == 0) & (test_labels == 1))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print("\n--- Test Evaluation ---")
    print(f"Threshold (95th percentile): {thresh:.5f}")
    print(f"Confusion Matrix -> TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

    np.savez("results.npz", errors=errors, labels=test_labels, preds=anomalies,
          threshold=thresh, precision=prec, recall=rec, f1=f1)


if __name__ == "__main__":
    main()