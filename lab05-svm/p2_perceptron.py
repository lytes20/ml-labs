import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class1 = np.array([[1, 1], [1, 2], [2, 1]])
class2 = np.array([[0, 0], [1, 0], [0, 1]])
X = np.vstack([class1, class2])
y = np.array([1, 1, 1, -1, -1, -1])

# SVM (from problem 1)
svm = SVC(kernel='linear', C=1e6).fit(X, y)
w_svm = svm.coef_[0]
b_svm = svm.intercept_[0]
m_svm = 1 / np.linalg.norm(w_svm)


# Classic Perceptron learning rule (Rosenblatt), zero-initialized weights,
# single pass repeated over the data in order until no mistakes remain.
def perceptron(X, y, eta=1.0, max_epochs=100):
    w = np.zeros(X.shape[1])
    b = 0.0
    for epoch in range(max_epochs):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (np.dot(w, xi) + b) <= 0:
                w += eta * yi * xi
                b += eta * yi
                errors += 1
        if errors == 0:
            return w, b, epoch + 1
    return w, b, max_epochs


w_p, b_p, n_epochs = perceptron(X, y)
m_p = min(abs(w_p[0] * x[0] + w_p[1] * x[1] + b_p) for x in X) / np.linalg.norm(w_p)

print("Perceptron converged after", n_epochs, "epochs")
print("w_perceptron =", w_p, "b_perceptron =", b_p)
print("Perceptron margin (dist. to closest point) =", m_p)
print("SVM margin =", m_svm)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(class1[:, 0], class1[:, 1], c='blue', marker='o', s=100, label='Class 1 (+1)')
ax.scatter(class2[:, 0], class2[:, 1], c='red', marker='s', s=100, label='Class 2 (-1)')

xx = np.linspace(-1, 3, 200)
yy_svm = -(w_svm[0] * xx + b_svm) / w_svm[1]
yy_p = -(w_p[0] * xx + b_p) / w_p[1]

ax.plot(xx, yy_svm, 'k-', linewidth=2, label='SVM boundary')
ax.plot(xx, yy_p, 'g--', linewidth=2, label='Perceptron boundary')

ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Problem 2: Perceptron vs. SVM')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, linestyle=':')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('p2_perceptron.png', dpi=150)
print("saved p2_perceptron.png")
