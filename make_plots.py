import numpy as np
import matplotlib.pyplot as plt

train_loss = np.array([0.1802, 0.1334, 0.1546, 0.177, 0.3095, 0.3283])
val_loss = np.array([0.5792, 0.3662, 0.2846, 0.2412, 0.2724, 0.2899])

train_compiled = np.array([24.45, 48.35, 53.17, 46.49, 46, 40.56])
val_compiled = np.array([44.29, 48.26, 53.02, 58.40, 43.14, 42.83])

train_tests = np.array([27.75, 42.59, 47.11, 50.11, 41.63, 36.50])
val_tests = np.array([44.29, 48.26, 53.02, 58.40, 43.14, 42.83])

plt.plot(train_loss, label="Train")
plt.plot(val_loss, label="Val")
plt.legend()
plt.savefig("Loss.png")
plt.show()


plt.plot(train_compiled, label="Train")
plt.plot(val_compiled, label="Val")
plt.legend()
plt.savefig("Compiled.png")
plt.show()


plt.plot(train_tests, label="Train")
plt.plot(val_tests, label="Val")
plt.legend()
plt.savefig("Tests.png")
plt.show()


