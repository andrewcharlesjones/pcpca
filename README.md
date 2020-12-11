# pcpca

Models and algorithms for probabilistic contrastive principal component analysis (PCPCA).

Given a pxn matrix X of foreground samples and a pxm matrix Y of background samples, PCPCA can be fit as follows.

```python
pcpca = PCPCA(pcpca = PCPCA(gamma=0.7, n_components=2)
pcpca.fit(X, Y)
```

Once the model is fit, samples can be projected onto the components by calling `transform`:

```python
pcpca.transform(X, Y)
```

Or both of these steps can be done with one call to `fit_transform`:

```python
pcpca.fit_transform(X, Y)
```
