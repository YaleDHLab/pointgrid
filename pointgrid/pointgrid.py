from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
import time, base64, math

def align_points_to_grid(arr, fill=0.1, pad=0.0, optimal_assignments=False, log_every=None):
  '''
  Snap each point in `arr` to the closest unoccupied slot in a mesh
  @arg arr numpy.ndarray:
    a numpy array with shape (n,2)
  @kwarg fill float:
    a number 0:1 that indicates what fraction of the grid slots should be
    filled with points
  @kwarg pad float:
    a decimal value 0:1 that indicates how much padding to add to the border
    domains to allow jitter positions to move beyond initial data domain
  @kwarg log_every integer:
    if a positive integer `n` is provided, the function informs the user every
    time `n` more points have been assigned grid positions
  @kwarg optimal_assignments bool:
    if True assigns each point to its closest open grid point, otherwise an
    approximately optimal open grid point is selected. True requires more
    time to compute
  @returns numpy.ndarray:
    with shape identical to the shape of `arr`
  '''
  if fill == 0: raise Exception('fill must be greater than 0 and less than 1')
  # create height and width of grid as function of array size and desired fill proportion
  h = w = math.ceil((len(arr)/fill)**(1/2))
  # find the bounds for the distribution
  bounds = get_bounds(arr, pad=pad)
  # create the grid mesh
  grid = create_mesh(arr, h=h, w=w, bounds=bounds)
  # fill the mesh
  print(' * filling mesh')
  df = pd.DataFrame(arr, columns=['x', 'y']).copy(deep=True)
  # store the number of points slotted
  c = 0
  for site, point in df.sample(frac=1)[['x', 'y']].iterrows():
    # skip points not in original points domain
    if point.y < bounds[0] or point.y > bounds[1] or \
       point.x < bounds[2] or point.x > bounds[3]:
      raise Exception('Input point is out of bounds', point.x, point.y, bounds)
    # initialize the search radius we'll use to slot this point in an open grid position
    r_y = (bounds[1]-bounds[0])/h
    r_x = (bounds[3]-bounds[2])/w
    slotted = False
    while not slotted:
      x, y = _get_grid_location(grid, point, r_x, r_y, optimal_assignments=optimal_assignments)
      # no open slots were found so increase the search radius
      if np.isnan(x):
        r_y *= 2
        r_x *= 2
      # success! optionally report the slotted position to the user
      else:
        # assign a value other than 1 to mark this slot as filled
        grid.loc[x, y] = 2
        df.loc[site, ['y', 'x']] = [x,y]
        slotted = True
        c += 1
        if log_every and c % log_every == 0:
          print(' * slotted', c, 'of', len(arr), 'assignments')
  return df.sort_index().to_numpy()

def get_bounds(arr, pad=0.2):
  '''
  Given a 2D array return the y_min, y_max, x_min, x_max
  @arg arr numpy.ndarray:
    a numpy array with shape (n,2)
  @kwarg pad float:
    a decimal value 0:1 that indicates how much padding to add to the border
    domains to allow jitter positions to move beyond initial data domain
  @returns list
    a list with [y_min, y_max, x_min, x_max]
  '''
  x_dom = [np.min(arr[:,1]), np.max(arr[:,1])]
  y_dom = [np.min(arr[:,0]), np.max(arr[:,0])]
  return [
    x_dom[0] - np.abs((x_dom[1]-x_dom[0])*pad),
    x_dom[1] + np.abs((x_dom[1]-x_dom[0])*pad),
    y_dom[0] - np.abs((y_dom[1]-y_dom[0])*pad),
    y_dom[1] + np.abs((y_dom[1]-y_dom[0])*pad),
  ]

def create_mesh(arr, h=100, w=100, bounds=[]):
  '''
  Given a 2D array create a mesh that will hold updated point positions
  @arg arr numpy.ndarray:
    a numpy array with shape (n,2)
  @kwarg h int:
    the number of unique height positions to create
  @kwarg w int:
    the number of unique width positions to create
  @kwarg bounds arr:
    a list with [y_min, y_max, x_min, x_max]
  @returns pandas.core.frame.DataFrame
     dataframe containing the available grid positions
  '''
  print(' * creating mesh with size', h, w)
  # create array of valid positions
  y_vals = np.arange(bounds[0], bounds[1], (bounds[1]-bounds[0])/h)
  x_vals = np.arange(bounds[2], bounds[3], (bounds[3]-bounds[2])/w)
  # create the dense mesh
  data = np.tile(
    [[0, 1], [1, 0]],
    np.array([
      int(np.ceil(len(y_vals) / 2)),
      int(np.ceil(len(x_vals) / 2)),
    ]))
  # ensure each axis has an even number of slots
  if len(y_vals) % 2 != 0 or len(x_vals) % 2 != 0:
    data = data[0:len(y_vals), 0:len(x_vals)]
  return pd.DataFrame(data, index=y_vals, columns=x_vals)

def _get_grid_location(grid, point, r_x, r_y, optimal_assignments=False):
  '''
  Find the x,y positions in `grid` to which `point` should be assigned
  @arg grid pandas.core.frame.DataFrame:
    dataframe containing the available grid positions
  @arg point tuple:
    a row from `grid` with x, y attributes
  @arg r_x float:
    the search radius to use in the x direction
  @arg r_y float:
    the search radius to use in the y direction
  @kwarg optimal_assignments bool:
    if True assigns each point to its closest open grid point, otherwise an
    approximately optimal open grid point is selected. True requires more
    time to compute
  @returns list
    the ideal [x,y] positions for `point` in `grid` if found, else
    [np.nan, np.nan]
  '''
  bottom = grid.index.searchsorted(point.y - r_y)
  top = grid.index.searchsorted(point.y + r_y, side='right')
  left = grid.columns.searchsorted(point.x - r_x)
  right = grid.columns.searchsorted(point.x + r_x, side='right')
  close_grid_points = grid.iloc[bottom:top, left:right]
  # if using optimal_assignments, store the position in this point's radius that minimizes distortion
  # else return the first open position within this point's current radius r_x, r_y
  best_dist = np.inf
  grid_loc = [np.nan, np.nan]
  for x, col in close_grid_points.iterrows():
    for y, val in col.items():
      if val != 1: continue
      if not optimal_assignments:
        return [x, y]
      else:
        dist = euclidean(point, (x,y))
        if dist < best_dist:
          best_dist = dist
          grid_loc = [x,y]
  return grid_loc