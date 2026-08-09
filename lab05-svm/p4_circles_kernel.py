import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.svm import SVC

rng = np.linspace(0, 2 * np.pi, 10, endpoint=False)

R_outer, R_inner = 3.0, 1.0
outer = np.column_stack([R_outer * np.cos(rng), R_outer * np.sin(rng)])
inner = np.column_stack([R_inner * np.cos(rng), R_inner * np.sin(rng)])

X = np.vstack([outer, inner])
y = np.array([1] * len(outer) + [-1] * len(inner))   # outer = +1, inner = -1

# ---- 2D plot: not linearly separable ----
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(outer[:, 0], outer[:, 1], c='blue', marker='o', s=80, label='Outer circle (+1)')
ax1.scatter(inner[:, 0], inner[:, 1], c='red', marker='s', s=80, label='Inner circle (-1)')
ax1.set_title('Original 2D data (not linearly separable)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.legend(fontsize=8)
ax1.grid(True, linestyle=':')
ax1.set_aspect('equal')

# ---- Polynomial kernel trick: explicit degree-2 feature map ----
# K(x, z) = (x . z)^2 corresponds to phi(x1, x2) = (x1^2, sqrt(2) x1 x2, x2^2)
def phi(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([x1 ** 2, np.sqrt(2) * x1 * x2, x2 ** 2])


X_3d = phi(X)

# sanity check: dot product in 3D feature space equals (x.z)^2 in 2D
i, j = 0, 5
lhs = np.dot(X_3d[i], X_3d[j])
rhs = np.dot(X[i], X[j]) ** 2
print(f"Kernel check: phi(x).phi(z) = {lhs:.4f},  (x.z)^2 = {rhs:.4f}")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
outer_3d = X_3d[y == 1]
inner_3d = X_3d[y == -1]
ax2.scatter(outer_3d[:, 0], outer_3d[:, 1], outer_3d[:, 2], c='blue', marker='o', s=60, label='Outer circle (+1)')
ax2.scatter(inner_3d[:, 0], inner_3d[:, 1], inner_3d[:, 2], c='red', marker='s', s=60, label='Inner circle (-1)')
ax2.set_title(r'Kernel-transformed data: $\phi(x_1,x_2)=(x_1^2,\sqrt{2}x_1x_2,x_2^2)$')
ax2.set_xlabel('z1 = x1^2')
ax2.set_ylabel('z2 = sqrt(2) x1 x2')
ax2.set_zlabel('z3 = x2^2')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('p4_circles_kernel.png', dpi=150)
print("saved p4_circles_kernel.png")

# ---- Optional (4b): run SVM with a polynomial kernel directly on the data ----
clf = SVC(kernel='poly', degree=2, coef0=0, gamma=1)
clf.fit(X, y)
preds = clf.predict(X)
print("\nSVM with polynomial kernel (degree=2) on original 2D data:")
print("Predictions:", preds)
print("True labels:", y)
print("Accuracy:", (preds == y).mean())
print("Number of support vectors:", len(clf.support_vectors_))

# Visualize the decision boundary in 2D
fig2, ax = plt.subplots(figsize=(6, 6))
xx, yy = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=['#ffcccc', '#cce5ff'], alpha=0.6)
ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
ax.scatter(outer[:, 0], outer[:, 1], c='blue', marker='o', s=80, label='Outer circle (+1)')
ax.scatter(inner[:, 0], inner[:, 1], c='red', marker='s', s=80, label='Inner circle (-1)')
ax.set_title('Problem 4b: SVM decision boundary (polynomial kernel)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(fontsize=8)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('p4b_svm_boundary.png', dpi=150)
print("saved p4b_svm_boundary.png")
