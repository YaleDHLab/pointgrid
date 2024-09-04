# Note: This repository has been archived
This project was developed under a previous phase of the Yale Digital Humanities Lab. Now a part of Yale Library’s Computational Methods and Data department, the Lab no longer includes this project in its scope of work. As such, it will receive no further updates.


# Pointgrid

> Transform a distribution of 2D points to a grid that preserves the global distribution shape. Useful for avoiding overplotting in data visualizations.

#### Before
![before](images/input.png 'before')

#### After
![after](images/output.png 'after')

## Installation

```bash
pip install pointgrid
```

## Usage

```python
from pointgrid import align_points_to_grid
from sklearn import datasets

# create fake data
arr, labels = datasets.make_blobs(n_samples=1000, centers=5)

# get updated point positions
updated = align_points_to_grid(arr)
```

`updated` will be a numpy array with the same shape as the input array `arr`.
