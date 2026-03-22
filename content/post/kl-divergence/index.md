---
title: 'KL Divergence'
subtitle: 'Entropy, KL Divergence and Cross Entropy in PyTorch'
summary: 'Entropy, KL Divergence and Cross Entropy in PyTorch'
authors:
- admin
tags:
- Information Theory
- Python
categories: []
date: "2020-09-01T00:00:00Z"
lastmod: ""
featured: false
draft: false

image:
  caption: ''
  focal_point: ""
  preview_only: false

projects: []
---

[Colab Notebook](https://colab.research.google.com/drive/1EXb4dW-jVaYsYOpYvLRZ5WVhivh-aglU)

Before seeing KL Divergence, let's see a very simple concept called Entropy

# Entropy

Entropy is the expected information contained in a Distribution. It measures the uncertainty.

$H(x) = \sum{p(x)I(x)}$ where $I(x)$ is called the Information content of $x$.

"If an event is very probable, it is no surprise (and generally uninteresting) when that event happens as expected. However, if an event is unlikely to occur, it has much more information to learn that the event happened or will happen. For instance, the knowledge that some particular number will not be the winning number of a lottery provides very little information, because any particular chosen number will almost certainly not win. However, the knowledge that a particular number will win a lottery has a high value because it communicates the outcome of a very low probability event."

So the information content is an increasing function of inverse of its probability. More the probability, lesser the information content and vice versa.
${\bf I(x) = log_2(\dfrac{1}{p(x)}) = -log_2(p(x))}$

Therefore,
${\bf H(x) = \sum{p(x)I(x)} = -\sum{p(x)log_2(p(x))}}$
and for continuous case ${\bf H(x) = -\int{p(x)log_2(p(x))}}$.

## Entropy in Pytorch

```python
import torch

def entropy(p, epsilon=1e-8):
    '''
    epsilon to avoid log(0) error
    '''
    logp = torch.log2(p + epsilon)
    e = torch.sum(-p*logp)
    return e

# probability of choosing [a, b, c, d]
p1 = torch.Tensor([0.1, 0.2, 0.4, 0.3])
p2 = torch.Tensor([0.9, 0.0, 0.0, 0.1])

print('Entropy of Distribution {} = {}'.format(p1.numpy(), entropy(p1)))
print('Entropy of Distribution {} = {}'.format(p2.numpy(), entropy(p2)))
```

```
Entropy of Distribution [0.1 0.2 0.4 0.3] = 1.846439242362976
Entropy of Distribution [0.9 0.  0.  0.1] = 0.4689956307411194
```

The first distribution has more entropy(uncertainty) as when we sample a data, there is a fair chance of the sampled data is from any of a, b, c, d.

But the second distribution is almost deterministic and has less uncertainty.

# Kullback-Leibler Divergence

KL Divergence measures the similarity of two probability distribution. Let $P$ and $Q$ be two probability distributions. For ex: Let $P$ be a Gaussian Distribution and $Q$ be Uniform.

The Kullback-Leibler Divergence is

${\bf D_{KL}(p(x)||q(x)) = \int_{-\infty}^{+\infty} p(x)ln{\dfrac{p(x)}{q(x)}}}$

For discrete distribution,

${\bf D_{KL}(p(x)||q(x)) = \sum p(x)ln{\dfrac{p(x)}{q(x)}}}$

## Properties of KL Divergence

- KL divergence is not a distance measure or "metric" measure.
- It is not symmetric ie: $D_{KL}(p(x)||q(x)) \neq D_{KL}(q(x)||p(x))$.
- It need not satisfy the triangular inequality.
- It is a non-negative measure ie: $D_{KL}(p(x)||q(x)) \geq 0$ and $D_{KL}(p(x)||q(x)) = 0$ if and only if $p(x)=q(x)$.

## KL Divergence in PyTorch

```python
import torch

def kl_divergence(p, q, epsilon=1e-8):
    '''
    epsilon to avoid log(0) or divided by 0 error
    '''
    d_kl = (p * torch.log((p / (q+ epsilon)) + epsilon)).sum()
    return d_kl

# comparison of kl of similar distribution p2-p3
# and dissimilar distribution p1-p2
p1 = torch.Tensor([0.1, 0.2, 0.4, 0.3])
p2 = torch.Tensor([0.9, 0.0, 0.0, 0.1])
p3 = torch.Tensor([0.8, 0.0, 0.05, 0.15])

print('KL Divergence of {} and {} = {}'.format(p1.numpy(), p1.numpy(), kl_divergence(p1, p2)))
print('KL Divergence of {} and {} = {}'.format(p2.numpy(), p3.numpy(), kl_divergence(p2, p3)))
```

```
KL Divergence of [0.1 0.2 0.4 0.3] and [0.1 0.2 0.4 0.3] = 10.47386646270752
KL Divergence of [0.9 0.  0.  0.1] and [0.8  0.   0.05 0.15] = 0.06545820087194443
```

# Mutual Information with KL Divergence

Let random variable $x$ and $y$ has distribution of $p(x)$ and $p(y)$.
If $x$ and $y$ are independent, then $p(x,y) = p(x)p(y)$.

If the distance between $p(x,y)$ and $p(x)p(y)$ becomes larger, $p(x)$ and $p(y)$ becomes dependent. We can use KL Divergence of $p(x,y)$ and $p(x)p(y)$ to measure the dependency of x and y.

${\bf I(x, y) = D_{KL}(p(x,y)||p(x)p(y)) = \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} p(x, y)ln{\dfrac{p(x, y)}{p(x)p(y)}}}$

# Cross Entropy

The Kullback-Leibler Divergence of $p(x)$ and $q(x)$ is given by

$D_{KL}(p(x)||q(x)) = \int_{-\infty}^{+\infty} p(x)ln{\dfrac{p(x)}{q(x)}}$

$\qquad = \int_{-\infty}^{+\infty} p(x)ln(p(x))-\int_{-\infty}^{+\infty}p(x)ln(q(x))$

$D_{KL}(p(x)||q(x)) = - H(p(x)) -\int_{-\infty}^{+\infty}p(x)ln(q(x))$

$D_{KL}(p(x)||q(x)) + H(p(x)) = -\int_{-\infty}^{+\infty}p(x)ln(q(x)) = H(p(x), q(x))$

$H(p(x), q(x))$ is called the Cross-Entropy. When $p(x)$ and $q(x)$ are same, then the Cross Entropy is equal to the entropy as $D_{KL}(p(x)||q(x))=0$.

The amount by which the cross entropy exceeds the entropy is called relative entropy or KL Divergence.

${\bf Cross\ Entropy = KL\ Divergence + Entropy }$

## Cross Entropy in PyTorch

```python
import torch

def cross_entropy(p, q, epsilon=1e-8):
    '''
    epsilon to avoid log(0) error
    '''
    logq = torch.log2(q + epsilon)
    ce = torch.sum(-p*logq)
    return ce

# comparison of cross_entropy of similar distribution p2-p3
# and dissimilar distribution p1-p2
p1 = torch.Tensor([0.1, 0.2, 0.4, 0.3])
p2 = torch.Tensor([0.9, 0.0, 0.0, 0.1])
p3 = torch.Tensor([0.8, 0.0, 0.05, 0.15])

print('Cross Entropy of {} and {} = {}'.format(p1.numpy(), p2.numpy(), cross_entropy(p1, p2)))
print('Cross Entropy of {} and {} = {}'.format(p2.numpy(), p3.numpy(), cross_entropy(p2, p3)))
```

```
Cross Entropy of [0.1 0.2 0.4 0.3] and [0.9 0.  0.  0.1] = 16.957033157348633
Cross Entropy of [0.9 0.  0.  0.1] and [0.8  0.   0.05 0.15] = 0.5634317994117737
```

## Reference

- Elements of Information Theory — Thomas M. Cover, Joy A. Thomas
- https://web.stanford.edu/~montanar/RESEARCH/BOOK/partA.pdf
- https://en.wikipedia.org/wiki/Cross_entropy
