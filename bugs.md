- [Callback error updating ..network-graph.elements...max-store.data..](#callback-error-updating-network-graphelementsmax-storedata)
  - [Traceback](#traceback)
  - [Steps to reproduce](#steps-to-reproduce)

-------

# Callback error updating ..network-graph.elements...max-store.data..

## Traceback

>Traceback (most recent call last):
  File "/Users/megan/Library/Mobile Documents/com~apple~CloudDocs/_Family/winter/winter-git/data.py", line 401, in make_graph_data
    tsne.fit_transform(coocc), index=tsne.feature_names_in_, columns=["x", "y"]
  File "/Users/megan/Library/Mobile Documents/com~apple~CloudDocs/_Family/winter/winter-git/.venv/lib/python3.9/site-packages/sklearn/manifold/_t_sne.py", line 1108, in fit_transform
    embedding = self._fit(X)
  File "/Users/megan/Library/Mobile Documents/com~apple~CloudDocs/_Family/winter/winter-git/.venv/lib/python3.9/site-packages/sklearn/manifold/_t_sne.py", line 830, in _fit
    X = self._validate_data(
  File "/Users/megan/Library/Mobile Documents/com~apple~CloudDocs/_Family/winter/winter-git/.venv/lib/python3.9/site-packages/sklearn/base.py", line 566, in _validate_data
    X = check_array(X, **check_params)
  File "/Users/megan/Library/Mobile Documents/com~apple~CloudDocs/_Family/winter/winter-git/.venv/lib/python3.9/site-packages/sklearn/utils/validation.py", line 805, in check_array
    raise ValueError(
ValueError: Found array with 1 sample(s) (shape=(1, 1)) while a minimum of 2 is required.

## Steps to reproduce

1. Restrict years to maximum `1990`
2. Select franchise `Nintendo`