import numpy as np
import matplotlib.pyplot as plt

data = np.load("results.npz")
errors = data["errors"]
labels_test = data["labels"]
predicted_anomaly = data["preds"]
threshold = data["threshold"]
precision = data["precision"]
recall = data["recall"]
f1 = data["f1"]

plt.subplot(2, 1, 1)
plt.plot(errors, label="Prediction error")
plt.axhline(threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.3f})")
plt.ylabel("Squared error")
plt.title("LSTM Prediction Error")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(labels_test, label="True label", color="black")
plt.plot(predicted_anomaly, label="Predicted", color="orange", linestyle="--")
plt.ylabel("Anomaly")
plt.xlabel("Time step")
plt.title(f"True vs Predicted (Prec={precision:.2f}, Rec={recall:.2f}, F1={f1:.2f})")
plt.legend()

plt.tight_layout()
plt.savefig("results_plot.png")