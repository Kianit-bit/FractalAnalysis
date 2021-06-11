import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join 

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def draw_clusters(data, clust, name):
  if data.shape[1] != 2:
    raise Exception('number of columns should be equal to 2. The number of columns was: {}'.format(data.shape[1]))
  plt.figure(figsize=(15,15))
  plt.title('draw_clusters')
  plt.scatter(data[:, 0], data[:,1], c = clust, s=6)
  plt.savefig("result/"+name[:-4]+"_clustered.png", dpi=300)
  plt.close()

def draw_ground_truth(data, parts, name):
  if data.shape[1] != 2:
    raise Exception('number of columns should be equal to 2. The number of columns was: {}'.format(data.shape[1]))
  size = math.ceil(math.sqrt(len(parts)))
  labels_pred = np.zeros(len(parts)).astype(int)
  patch_size = size//2
  for i in range(0, size, 2):
    labels_pred[i*patch_size:(i+1)*patch_size] = 0
  for i in range(1, size, 2):
    labels_pred[i*patch_size:(i+1)*patch_size] = 1
  for i in range(size, size*2, 2):
    labels_pred[i*patch_size:(i+1)*patch_size] = 2
  for i in range(size+1, size*2, 2):
    labels_pred[i*patch_size:(i+1)*patch_size] = 3

  plt.figure(figsize=(15,15))
  plt.title('ground_truth')
  plt.scatter(data[:, 0], data[:,1], c = labels_pred, s=6)
  plt.savefig("result/"+name[:-4]+"_ground_truth.png", dpi=300)
  plt.close()

def describe(ads, spec, name, parts):
  ads = pd.DataFrame(ads)
  spec = pd.DataFrame(spec)  

  ####
  size = math.ceil(math.sqrt(len(parts)))
  labels_pred = np.zeros(len(parts)).astype(int)
  patch_size = size//2
  for i in range(0, size, 2):
    labels_pred[i*patch_size:(i+1)*patch_size] = 0
  for i in range(1, size, 2):
    labels_pred[i*patch_size:(i+1)*patch_size] = 1
  for i in range(size, size*2, 2):
    labels_pred[i*patch_size:(i+1)*patch_size] = 2
  for i in range(size+1, size*2, 2):
    labels_pred[i*patch_size:(i+1)*patch_size] = 3

  first = np.argwhere(labels_pred == 0).reshape(1,-1)[0]
  second = np.argwhere(labels_pred == 1).reshape(1,-1)[0]
  third = np.argwhere(labels_pred == 2).reshape(1,-1)[0]
  fourth = np.argwhere(labels_pred == 3).reshape(1,-1)[0]

  ####
  bla = []
  bla.append( [ads.iloc[first].mean(), ads.iloc[first].std()] )
  bla.append( [ads.iloc[second].mean(), ads.iloc[second].std()] )
  bla.append( [ads.iloc[third].mean(), ads.iloc[third].std()] )
  bla.append( [ads.iloc[fourth].mean(), ads.iloc[fourth].std()] )

  bla2 = []
  bla2.append( [spec.iloc[first].mean(), spec.iloc[first].std()] )
  bla2.append( [spec.iloc[second].mean(), spec.iloc[second].std()] )
  bla2.append( [spec.iloc[third].mean(), spec.iloc[third].std()] )
  bla2.append( [spec.iloc[fourth].mean(), spec.iloc[fourth].std()] )

  labels = ['upper left','upper right','lower left','lower right']
  colors = ['red', 'green', 'blue', 'yellow']
  i=0
  plt.figure(figsize=(10,10))
  for elem in bla:
    x = range(0, len(elem[0]))
    plt.scatter(x, elem[0], label=labels[i], c = colors[i])
    data = {
        'x': x,
        'y1': [y - 2*e for y, e in zip(elem[0], elem[1])],
        'y2': [y + 2*e for y, e in zip(elem[0], elem[1])]
    }
    plt.fill_between(**data, alpha=.25, color=colors[i])    
    plt.legend()
    i += 1
  plt.savefig("result/" + name[:-4] + "_mfs" + ".png", dpi=300)
  plt.close()

  i=0
  plt.figure(figsize=(10,10))
  for elem in bla2:
    #x = np.array([1,2,3,4,5,7,10,15,20,30,40])
    x = np.array([-20,-15,-10,-5,-4,-3,-2,-1, 1, 2,4, 7, 10, 15, 20]) 
    x = range(x[0], x[-1]+1)    
    plt.scatter(x, elem[0], label=labels[i], c=colors[i])    
    data = {
        'x': x,
        'y1': [y - e for y,e in zip(elem[0], elem[1])],
        'y2': [y + e for y,e in zip(elem[0], elem[1])]
    }
    plt.fill_between(**data, alpha=.25, color=colors[i])
    plt.legend()
    i += 1
  plt.savefig("result/" + name[:-4] + "_reni" + ".png", dpi=300)
  plt.close()

#region draw_gmm
#https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
#endregion