# Lab 4 – Dimensionality Reduction and MLP — Solutions

## 1. Why does unit hypersphere volume decrease (rather than increase) as the number of dimensions exceeds 5?

The volume of an n-dimensional unit hypersphere (radius R = 1) is given by:

V(n) = π^(n/2) / Γ(n/2 + 1)

As n grows, the Gamma function in the denominator grows *faster* than the π^(n/2) term in the numerator (factorial-like growth beats exponential growth), so V(n) rises for small n, peaks around n = 5, and then shrinks toward 0 as n → ∞.

The intuitive reason behind this is:

- A point is "inside" the unit hypersphere only if **all n of its coordinates are simultaneously small enough** that their squared sum stays under 1. Every additional dimension adds one more constraint that must be satisfied at the same time.
- Think of the hypersphere as living inside its bounding hypercube of side length 2. As dimensions increase, more and more of the hypercube's volume gets pushed out into its "corners" — regions where coordinates are all near their extreme values — while the sphere only covers the rounded middle region. Since a hypercube's corner-to-total-volume ratio increases with dimension, the sphere (which excludes the corners) ends up capturing a shrinking fraction of the hypercube.
- Equivalently, each new dimension multiplies the "probability of staying inside" by another factor less than 1, so the volume shrinks geometrically once dimensionality is high enough.

This is one of the classic illustrations of the **curse of dimensionality**: our low-dimensional geometric intuition (spheres "fill up" the space they're inscribed in) breaks down as dimensionality grows — in high dimensions, almost all the volume lives in a thin shell/corners far from the center, and data points become increasingly sparse and spread apart, which is exactly why dimensionality reduction becomes important for high-dimensional ML problems.

## 2. Summary of Paper (a): "Analysis of Dimensionality Reduction Techniques on Big Data"

*Reddy, G.T. et al., IEEE Access, vol. 8, pp. 54776–54788, 2020.*

This paper studies how two classic linear dimensionality-reduction (DR) techniques — **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** — affect the performance of downstream machine learning classifiers on big-data-style datasets. The authors apply PCA and LDA as a preprocessing step before training four classifiers (**Decision Tree, SVM, Naive Bayes, and Random Forest**) on three real-world datasets with differing dimensionality (Cardiotocography, Diabetic Retinopathy, and an Intrusion Detection dataset).

The core motivation is practical: not every attribute in a large dataset is actually useful for prediction, and irrelevant/redundant features increase both computational cost and the risk of overfitting. Removing them via DR should, in principle, make training faster and models more generalizable.

Their key findings were:

- **PCA consistently outperformed LDA** across the evaluation metrics used, suggesting that unsupervised variance-preserving projection generalized better than the label-driven LDA approach for these datasets.
- **Tree-based models (Decision Tree, Random Forest) were largely insensitive** to whether DR was applied at all — their accuracy didn't change much with or without PCA/LDA, likely because tree splits already perform an implicit form of feature selection.
- **DR's benefit is data-dependent, not universal**: applying PCA helped the most when the original dataset had high dimensionality, but on datasets that were already low-dimensional, skipping DR and training directly on the raw features produced better results.

My takeaway from the paper is that dimensionality reduction should be treated as a *hyperparameter/design choice to be validated*, not a step to apply blindly — its usefulness depends heavily on how many original features you have and which downstream model you're using.

*(Paper (b), Cunningham & Ghahramani's "Linear Dimensionality Reduction: Survey, Insights, and Generalizations," is the more mathematically dense unifying-framework paper referenced in the assignment for further reading; per the instructions, only paper (a) was required to be summarized above.)*

## 3. Key points I can apply to my project

> The specific relevance below depends on the dataset and model used in my project — replace/tailor the details as needed before submitting.

A few concrete takeaways from the paper that are broadly applicable:

1. **Don't assume DR helps — test it.** The paper shows that on low-dimensional data, skipping PCA/LDA outperformed applying it. Before adding a DR step to a pipeline, I should benchmark model performance with and without it rather than assuming it will improve results.
2. **Match the DR technique to the model.** Since tree-based models (Decision Tree, Random Forest) were largely unaffected by DR while other classifiers (SVM, Naive Bayes) were more sensitive, the choice of downstream model should inform whether DR is worth the extra preprocessing step.
3. **Prefer PCA as a default over LDA** when there isn't a strong reason to use label information for the projection, based on this paper's result that PCA generalized better across datasets.

If my current project doesn't yet involve high-dimensional data, DR is still likely to become relevant as the project scales — for example:

- If I move from a small hand-engineered feature set to something like **text embeddings, image pixel data, or dense sensor/IoT streams**, feature counts can jump from tens to hundreds or thousands of dimensions.
- At that point, the curse-of-dimensionality effect from Question 1 kicks in: distances between points become less meaningful, data becomes sparse, and models need proportionally more data to generalize well.
- Applying PCA (or LDA, if labels are available and relevant) at that stage would help control training time, reduce overfitting risk, and make visualization of the data (e.g., 2D/3D projections) feasible again.

In short, DR isn't something my project needs right now, but it's a tool I'd reach for as soon as feature dimensionality grows large enough that training cost, overfitting, or the curse of dimensionality starts to become a bottleneck.
