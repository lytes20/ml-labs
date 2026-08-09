import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Data from problem 1
class1 = np.array([[1, 1], [1, 2], [2, 1]])   # label +1
class2 = np.array([[0, 0], [1, 0], [0, 1]])   # label -1

X = np.vstack([class1, class2])
y = np.array([1, 1, 1, -1, -1, -1])

clf = SVC(kernel='linear', C=1e6)
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]
margin = 1 / np.linalg.norm(w)

print("w =", w)
print("b =", b)
print("Decision boundary: {:.4f}*x1 + {:.4f}*x2 + {:.4f} = 0".format(w[0], w[1], b))
print("Margin (half-width, 1/||w||) =", margin)
print("Full margin (2/||w||) =", 2 * margin)
print("Support vectors:\n", clf.support_vectors_)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(class1[:, 0], class1[:, 1], c='blue', marker='o', s=100, label='Class 1 (+1)')
ax.scatter(class2[:, 0], class2[:, 1], c='red', marker='s', s=100, label='Class 2 (-1)')
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=250,
           facecolors='none', edgecolors='black', linewidths=2, label='Support vectors')

xx = np.linspace(-1, 3, 200)
yy = -(w[0] * xx + b) / w[1]
margin_up = yy + margin * np.sqrt(1 + (w[0] / w[1]) ** 2)
margin_down = yy - margin * np.sqrt(1 + (w[0] / w[1]) ** 2)

ax.plot(xx, yy, 'k-', label='Separating hyperplane')
ax.plot(xx, margin_up, 'k--', linewidth=1)
ax.plot(xx, margin_down, 'k--', linewidth=1)

for pt in X:
    ax.annotate(f"({pt[0]},{pt[1]})", (pt[0], pt[1]), textcoords="offset points", xytext=(8, 8))

ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Problem 1: SVM Optimal Separating Hyperplane')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, linestyle=':')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('p1_svm.png', dpi=150)
print("saved p1_svm.png")
