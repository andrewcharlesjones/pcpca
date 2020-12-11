# pcpca

Models and algorithms for probabilistic contrastive principal component analysis (PCPCA).

Given a $p \times n$ matrix $X$ of foreground samples and a $p \times m$ matrix $Y$ of background samples, PCPCA can be fit as follows.

```python
pcpca = PCPCA(pcpca = PCPCA(gamma=0.7, n_components=2)
pcpca.fit(X, Y)
```
