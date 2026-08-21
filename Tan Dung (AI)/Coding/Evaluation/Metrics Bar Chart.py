import matplotlib.pyplot as plt

metrics = {
    "Accuracy":0.7023,
    "Precision":1.0000,
    "Recall":0.6486,
    "F1":0.7869
}

plt.figure(figsize=(8,5))

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0,1.05)

plt.title("Experiment 1 - Performance Metrics")

plt.savefig("exp1_metrics.png")

plt.show()