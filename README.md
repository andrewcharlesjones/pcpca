# Probabilistic contrastive principal component analysis

This repo contains models and algorithms for probabilistic contrastive principal component analysis (PCPCA).

The accompanying paper can be found here: https://arxiv.org/abs/2012.07977.

## Installation

After cloning this repo, navigate to its directory, and run the following command.
```
python setup.py install
```

## Motivation

Given a foreground dataset X and a backround dataset Y, PCPCA is designed to find structure and variation that is enriched in the foreground relative to the background.

## Example

Given a p by n matrix X of foreground samples and a p by m matrix Y of background samples, PCPCA can be fit as follows.

```python
pcpca = PCPCA(gamma=0.7, n_components=2)
pcpca.fit(X, Y)
```

Once the model is fit, samples can be projected onto the components by calling `transform`:

```python
X_reduced, Y_reduced = pcpca.transform(X, Y)
```

Or both of these steps can be done with one call to `fit_transform`:

```python
X_reduced, Y_reduced = pcpca.fit_transform(X, Y)
```
