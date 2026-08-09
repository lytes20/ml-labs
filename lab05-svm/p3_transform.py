import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Data from problem 3
pos = np.array([[2, 2], [2, -2], [-2, -2], [-2, 2]])   # positive labels
neg = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])   # negative labels

X = np.vstack([pos, neg])
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])


def phi1(x1, x2):
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    if r > 2:
        d = abs(x1 - x2)
        return np.array([4 - x2 + d, 4 - x1 + d])
    else:
        return np.array([x1, x2])


X_t = np.array([phi1(x1, x2) for x1, x2 in X])

print("Original points -> Transformed points")
for orig, t, label in zip(X, X_t, y):
    print(f"  {tuple(orig)} (label {label:+d}) -> {tuple(t)}")

# Original (nonlinear) data plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes[0]
ax.scatter(pos[:, 0], pos[:, 1], c='blue', marker='o', s=100, label='Positive (+1)')
ax.scatter(neg[:, 0], neg[:, 1], c='red', marker='s', s=100, label='Negative (-1)')
circle = plt.Circle((0, 0), 2, color='gray', fill=False, linestyle=':')
ax.add_patch(circle)
for pt in X:
    ax.annotate(f"({pt[0]},{pt[1]})", (pt[0], pt[1]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_title('Original data (not linearly separable)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(fontsize=8)
ax.grid(True, linestyle=':')
ax.set_aspect('equal')

# Transformed data + linear classifier
clf = SVC(kernel='linear', C=1e6).fit(X_t, y)
w = clf.coef_[0]
b = clf.intercept_[0]
margin = 1 / np.linalg.norm(w)
print("\nTransformed-space SVM:")
print("w =", w, "b =", b, "margin =", margin)
print("Support vectors (transformed space):\n", clf.support_vectors_)

ax = axes[1]
pos_t = X_t[y == 1]
neg_t = X_t[y == -1]
ax.scatter(pos_t[:, 0], pos_t[:, 1], c='blue', marker='o', s=100, label='Positive (+1)')
ax.scatter(neg_t[:, 0], neg_t[:, 1], c='red', marker='s', s=100, label='Negative (-1)')
for pt in X_t:
    ax.annotate(f"({pt[0]:.0f},{pt[1]:.0f})", (pt[0], pt[1]), textcoords="offset points", xytext=(6, 6), fontsize=8)

xx = np.linspace(X_t[:, 0].min() - 1, X_t[:, 0].max() + 1, 200)
yy = -(w[0] * xx + b) / w[1]
ax.plot(xx, yy, 'k-', label='Separating line')

ax.set_title(r'Transformed data $\Phi_1(x_1,x_2)$ (linearly separable)')
ax.set_xlabel('z1')
ax.set_ylabel('z2')
ax.legend(fontsize=8)
ax.grid(True, linestyle=':')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('p3_transform.png', dpi=150)
print("saved p3_transform.png")
