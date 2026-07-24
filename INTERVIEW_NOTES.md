# Interview Notes — Classical ML Pivots

Use with `make demo` and the notebooks. Keep answers short and equation-first.

## Gradient descent
Update: `θ ← θ − η ∇L(θ)`.  
Batch = full gradient (stable, slow). SGD = one sample (noisy, fast). Mini-batch = compromise.

## Regularization
- **L2 (Ridge):** shrinks weights, keeps all features; stabilizes correlated inputs.  
- **L1 (Lasso):** promotes sparsity → feature selection.  
Never regularize the intercept.

## Bias–variance
Low degree → high bias / underfit. High degree → high variance / overfit.  
Cross-validation picks the sweet spot. Regularization reduces variance.

## Trees vs linear models
Trees: non-linear axis-aligned splits, handle interactions, overfit if deep.  
Linear/logistic: strong when decision boundary is roughly linear; calibrate with ROC-AUC.

## ROC / AUC
ROC plots TPR vs FPR over thresholds. AUC = ranking quality (probability a random positive scores above a random negative).

## SVM
Maximize margin; soft-margin uses hinge loss `max(0, 1 − y(w·x+b))`.  
Kernel SVM for non-linear boundaries (not implemented here — discuss RBF mentally).

## PCA vs KMeans
PCA: unsupervised linear compression maximizing variance.  
KMeans: partition by distance to centroids; sensitive to scale and init.

## Naive Bayes / KNN
NB: assumes feature independence given class — surprisingly strong text baseline.  
KNN: non-parametric; needs scaling; costly at inference.

## MLP backprop (say this)
Forward: `a¹ = ReLU(XW¹+b¹)`, `p = softmax(a¹W²+b²)`.  
Loss: cross-entropy. Backward: `∂L/∂z² = (p−y)/n`, then chain rule through ReLU mask.

## Attention
`Attention(Q,K,V) = softmax(QKᵀ/√d_k) V`.  
Self-attention: Q,K,V are projections of the same sequence. Softmax makes weights a distribution over keys.

## Calibration
High accuracy ≠ calibrated probabilities. Brier score and ECE quantify “when I say 80%, am I right 80% of the time?”

## Gradient boosting
Sequentially fit models to residuals / pseudo-residuals of a loss (here: logistic). Shrinkage (`learning_rate`) controls variance.

