# DeepLearning

Implementation of some very basic deep learning algorithms, in python and in 
C++, for my own practice and learning.

## Basic premise of learning algorithms

Let $\mathcal{X}\simeq\mathbb{R}^m$ be the feature space. Given a matrix 
$X\in\mathrm{M}_{N,m}(\mathbb{R})$ whose rows comprise a set of observations 
$x_1,\ldots,x_N\in \mathcal{X}$, and labels $y_1,\ldots, y_N\in \mathbb{R}$. 
(The labels can, and indeed will, be vectors of length $>1$; for simplicity
we'll postpone this discussion for later.)