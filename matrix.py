import matplotlib.pyplot as plt  # installed automatically with seaborn
import seaborn as sns
import numpy as np

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

# Use your already printed confusion matrix directly
cm = np.array([
    [6042, 1323,   95,    0,    0],
    [  81, 9415,  207,    5,    3],
    [ 163,  473, 1785,    0,    0],
    [   0, 2429,   17,  432,    7],
    [   0,   40,    3,    5,   19]
])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=150)

plt.show()