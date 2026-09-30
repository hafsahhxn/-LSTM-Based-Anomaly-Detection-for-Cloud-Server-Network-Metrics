import json
import time
import numpy as np
from data_loader import load_labels, load_series, normalize, make_sequences
from lstm import LSTM

DATA_PATH = "data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv"
FILE_KEY = "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv"
LABELS_PATH = "data/labels/combined_windows.json"

# Load data once
labels_dict = load_labels(LABELS_PATH)
df = load_series(DATA_PATH, FILE_KEY, labels_dict)
values = df["value"].values.astype(float)
norm_vals, _, _ = normalize(values)


def run_test(win_size, hidden_dim, lr, epochs, percentile):
    X, y, idx = make_sequences(norm_vals, win_size)
    labels = df["label"].values[idx]

    split = int(len(X) * 0.6)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    test_labels = labels[split:]

    model = LSTM(input_size=1, hidden_size=hidden_dim, output_size=1)

    t0 = time.time()
    for ep in range(epochs):
        losses = [model.train_step(X_train[i], y_train[i], lr) for i in range(len(X_train))]
    train_time = time.time() - t0

    preds = np.array([model.predict(x)[0] for x in X_test])
    errors = (preds - y_test.flatten()) ** 2
    thresh = np.percentile(errors, percentile)
    anomalies = (errors > thresh).astype(int)

    tp = np.sum((anomalies == 1) & (test_labels == 1))
    fp = np.sum((anomalies == 1) & (test_labels == 0))
    fn = np.sum((anomalies == 0) & (test_labels == 1))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "win": win_size,
        "hidden": hidden_dim,
        "lr": lr,
        "epochs": epochs,
        "percentile": percentile,
        "time": round(train_time, 2),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


# Run variations
configs = [
    (20, 16, 0.01, 5, 95),
    (20, 16, 0.01, 5, 90),
    (20, 16, 0.01, 5, 98),
    (10, 16, 0.01, 5, 95),
    (20, 32, 0.01, 5, 95)
]

results = []
for win, hid, lr, ep, pct in configs:
    print(f"Testing win={win}, hidden={hid}, pct={pct}...")
    res = run_test(win, hid, lr, ep, pct)
    results.append(res)
    print(res)

with open("experiment_log.json", "w") as f:
    json.dump(results, f, indent=4)