import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import os
import time
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool
from scipy.stats import linregress
from scipy.interpolate import interp1d
from PIL import Image
from scipy.ndimage.filters import convolve
from os import listdir
from os.path import isfile, join 
from scipy.ndimage.filters import maximum_filter, minimum_filter, generic_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn import metrics

from feature_dist import *
from clust_plot import *
from autoencoder import *
import fuzzy_cmeans as fm

def reduce_features(data, name, n_components=2):
  if (name == 'tsne'):
    m = TSNE(n_components=n_components, perplexity=30, n_iter=1000).fit_transform(data) # perplexity default = 30
  elif (name == 'pca'):
    m = PCA(n_components=0.95).fit_transform(data)
  elif (name == 'FA'):
    m = FactorAnalysis(n_components=n_components).fit_transform(data)
  else:
    m = TSNE(n_components=n_components).fit_transform(data)
  return m

def hubert_statistic():
    dist_points_from_cluster_center = []
    K = range(2,10)
    for no_of_clusters in K:
      k_model = KMeans(n_clusters=no_of_clusters)
      k_model.fit(data)

      c=k_model.labels_.astype(int)

      n = data.shape[0]
      distance_matrix = pairwise_distances(data, data, metric='euclidean')
      Q = np.zeros((n,n))
      clust_centers = k_model.cluster_centers_
      for i in range(n):
        for j in range(n):
          #Q[i][j] = distance.euclidean(clust_centers[c[i]], clust_centers[c[j]])
          Q[i][j] = np.linalg.norm(clust_centers[c[i]] - clust_centers[c[j]])

      norm_gamma = 0
      P_mean = np.mean(distance_matrix)
      Q_mean = np.mean(Q)
      P_var = np.var(distance_matrix)
      Q_var = np.var(Q)
      for i in range(n-1):
        for j in range(i+1,n):
          norm_gamma += (distance_matrix[i][j] - P_mean) * (Q[i][j] - Q_mean)
      norm_gamma /= P_var * Q_var
      norm_gamma /= (n * (n-1) / 2)

      dist_points_from_cluster_center.append(norm_gamma)

def find_best_k(data, alg, elem, mode='sil'):
    scores = []
    K = range(2,10)
    for no_of_clusters in K:
      if alg == 'aggl':
        distance_matrix = pairwise_distances(data, data, metric='euclidean')
        clustering = AgglomerativeClustering(n_clusters=no_of_clusters, affinity='precomputed', \
                                           linkage = 'average').fit(distance_matrix)
        scores.append(metrics.silhouette_score(data, clustering.labels_, metric='euclidean'))
      elif alg == 'cmeans':
        c, centers = fm.fcm(data, no_of_clusters)
        c = np.argmax(c,axis=0)
        scores.append(metrics.silhouette_score(data, c, metric='euclidean'))
      else:                    
        clustering = GaussianMixture(n_components=no_of_clusters, random_state=0, covariance_type='full').fit(data)
        if mode == 'bic':
            scores.append(-clustering.bic(data))
        else:
            scores.append(metrics.silhouette_score(data, clustering.predict(data), metric='euclidean'))
    plt.plot(K, scores)
    plt.savefig("result/" + elem[:-4]+ "_silhouette.png")
    plt.close()
    return np.argmax(scores) + 2

def assess(parts, c):
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

  val = adjusted_rand_score(c, labels_pred)
  print("adjusted_rand_score: ", val)
  return val

#factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0-contrast))
def align_range(region, factor):
    for pixel_ind in len(region):
        temp = (factor*(region[pixel_ind//2, pixel_ind%2] - 128) + 128)
        temp = 0 if temp < 0 else temp
        region[pixel_ind//2, pixel_ind%2] = 255 if temp > 255 else temp
    return region

def cluster(data, k, filename, parts, alg, eps):
  data = np.array(data)
  if alg == 'aggl':
      distance_matrix = pairwise_distances(data, data, metric='euclidean')
      clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', \
                                           linkage = 'average').fit(distance_matrix)
      # clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(ads)
      # plot_dendrogram(clustering, truncate_mode='level', p=30)
      c = clustering.labels_.astype(int)
  elif alg == 'cmeans':
      c, centers = fm.fcm(data, k, verbose=1)
      c = np.transpose(c)
      outliers = np.apply_along_axis(lambda arr: 1 - np.prod([1-a for a in arr]), 1, c)
      outliers = np.where(outliers < 0.4, True, False)
      c = np.argmax(c, axis=1)
      c[outliers] = 99
  else:
      gm = GaussianMixture(n_components=k, random_state=0, covariance_type='full').fit(data)
      matr = gm.predict_proba(data)
      outliers = np.apply_along_axis(lambda arr: 1 - np.prod([1-a for a in arr]), 1, matr)
      outliers = np.where(outliers < 0.4, True, False)
      c = np.argmax(matr, axis=1)
      c[outliers] = 99
      #c = gm.predict(ads).astype(int)
      if data.shape[1] == 2:
        plot_gmm(gm, data)
        plt.savefig("result/" + filename[:-4] + "gmm.png")  
  colors = [(0, 250, 0), (255,0,0), (0,255,255), (255,0,255), (255,255,51), \
	          (255,128,0), (255,102,255), (50,50,255), (125,0,125)]
  output = cv.imread("data/" + filename, cv.COLOR_BGR2RGB) 
  alpha=0.4
  overlay = output.copy()
  for i in range(0, len(parts)):
    if c[i] == 99:
      continue
    cv.circle(overlay, (parts[i][0]+eps//2, parts[i][1]+eps//2), 5, colors[c[i]], -1)
  cv.addWeighted(overlay, alpha, output, 1 - alpha,
		0, output)
  cv.imwrite("result/" + filename, output)  
  return c

#region mfs
def modified_signature(region, iter_count=45):
    #region = align_range(region)
    upper = region.copy()
    lower = region.copy()

    volumes = []
    mask=np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]]) 
    iter_range = range(1, iter_count)    

    for d in iter_range:
        scnd_u = maximum_filter(upper, mode='mirror', footprint=mask)
        scnd_b = minimum_filter(lower, mode='mirror', footprint=mask)
        upper = np.maximum(upper + 1, scnd_u)
        lower = np.minimum(lower - 1, scnd_b)
        volumes.append(np.sum(upper - lower))

    x = np.log2(iter_range[1:iter_count-2]) #or negative
    y = [np.log2((volumes[i] - volumes[i-1])/2) for i in range(1, iter_count-2)]   

    F_d = []
    for i in range(1, len(x)):
      F_d.append(2 - ((y[0]- y[i]) / (x[0]- x[i]))) 
    
    x = np.arange(2, iter_count-2)
    if (len(x) != len(F_d)):
      print("Length error")
    return F_d
#endregion

#region Renie
def integral_sum(bright_data):   
    integral_data = np.zeros(bright_data.shape)

    for y in range(bright_data.shape[0]):
        for x in range(bright_data.shape[1]):
            integral_data[y,x] = bright_data[y, x]

            if(y>0 and x>0):
                integral_data[y,x] +=  integral_data[y, x-1]
                integral_data[y,x] +=  integral_data[y-1,x]
                integral_data[y,x] -=  integral_data[y-1,x-1]
            else:
                if(y>0):
                    integral_data[y,x] +=  integral_data[y-1,x]
                if(x>0):                               
                    integral_data[y,x] +=  integral_data[y,x-1] 
    return integral_data

def calc_renie_entropy(p, q):
    return (1 / (1 - q) * np.log(np.sum(np.power(p, q)))) if q != 1 else (-np.sum(p * np.log(p)))

def calc_renie_dim(integ_sum, q, eps):
    cell_lengths = [2,3,4,6,8] 
    entr_vals = []
    for w in cell_lengths:
        conv = []
        for dx in range(0, 0+eps-w, w):
          for dy in range(0, 0+eps-w, w):
            conv.append(integ_sum[dy+w,dx+w] + integ_sum[dy,dx] - integ_sum[dy+w,dx] - integ_sum[dy,dx+w])
        entr_vals.append(calc_renie_entropy(conv / np.sum(conv), q))
    return linregress(-np.log(cell_lengths), entr_vals).slope

def calc_generalized_spectre(integ_sum, qs, eps):
    return list(map(lambda x: calc_renie_dim(integ_sum, x, eps), qs))

def calc_renie(integ_sum):
    eps = 24
    #q = np.array([1,2,3,4,5,7,10,15,20,30,40])
    q = np.array([-20,-15,-10,-5,-4,-3,-2,-1, 1, 2,4, 7, 10, 15, 20]) 
    elem = calc_generalized_spectre(integ_sum, q, eps)
    x = range(q[0], q[-1]+1)
    y = interp1d(q, elem, kind='cubic')(x)
    return y
#endregion

def Calc(filename, eps, rgb_or_hsv='rgb'):
  if rgb_or_hsv == 'rgb':
    img = cv.imread("data/" + filename, cv.IMREAD_GRAYSCALE)
    img_bright = np.array(img, dtype=np.float64)
  else:
    img = cv.imread("data/" + filename) # 
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_bright = np.array(img[:,:,2], dtype=np.float64)
    #value 0-100 or 0-1
    #v in hsv is equal to bright, if we use grayscale image

  parts = []
  ads10 = []
  img_pieces = []
  stepBetweenPixels = eps//2

  for start_y, end_y in zip(range(0, img_bright.shape[0]-eps+1, stepBetweenPixels), range(eps, img_bright.shape[0]+1, stepBetweenPixels)):
    for start_x, end_x in zip(range(0, img_bright.shape[1]-eps+1, stepBetweenPixels), range(eps, img_bright.shape[1]+1, stepBetweenPixels)):
      parts.append((start_x, start_y))
     
  ##############
  center_x, center_y = img_bright.shape[0] // 2, img_bright.shape[1] // 2
  parts = list(filter(lambda x: math.fabs(x[0]-center_x) > eps and math.fabs(x[1]-center_y) > eps and x[0]>=eps and x[1]>=eps, parts))
  ##############

  """Signature"""
  for part in parts:
    start_x, start_y = part
    img_pieces.append(img_bright[start_y:start_y+eps, start_x:start_x+eps])

  t1_start = time.perf_counter() 
  with Pool(4) as p:
    ads10 = p.map(modified_signature, img_pieces)
  t2_start = time.perf_counter() 
  print("MFS: ", int(t2_start - t1_start)//60, " min", (t2_start - t1_start) % 60, " sec")

  """Renie"""
  immat = img_bright.copy() * (254/255) + 1
  specs20 = []
  piece_integ_sum = []
  integ_sum = integral_sum(immat)

  #########################
  #print("sum of brightness 1: ", integ_sum[528,528] + integ_sum[0,0] - integ_sum[0,528]-integ_sum[528,0])
  #print("sum of brightness 2: ", integ_sum[528,1055] + integ_sum[0,528] - integ_sum[528,528]-integ_sum[0,1055])
  #print("sum of brightness 3: ", integ_sum[1055,528] + integ_sum[528,0] - integ_sum[1055,0]-integ_sum[528,528])
  #print("sum of brightness 4: ", integ_sum[1055,1055] + integ_sum[528,528] - integ_sum[1055,528]-integ_sum[528,1055])
  #########################

  for part in parts:
    start_x, start_y = part
    piece_integ_sum.append(integ_sum[start_y:start_y+eps, start_x:start_x+eps])

  t1_start = time.perf_counter() 
  with Pool(4) as p:
    specs20 = p.map(calc_renie, piece_integ_sum)
  t2_start = time.perf_counter() 
  print("Renie: ", int(t2_start - t1_start)//60, " min", (t2_start - t1_start) % 60, " sec")
  specs20 = np.array(specs20)
 
  return ads10, specs20, parts

if __name__ == '__main__':
    path = os.getcwd() 
    adj_scores = []
    onlyfiles = [f for f in listdir(path+ "/data") if isfile(join(path+ "/data", f))]
    
    f = open('scores.txt', 'w')
    f.close()

    for elem in onlyfiles:
        print("current image: " + elem)
        #get features
        eps = 24
        mfs_arr, renie_arr, parts = Calc(elem, eps, 'rgb')
        describe(mfs_arr, renie_arr, elem, parts)
        #distrib_plot(preprocessing.scale(mfs_arr), elem, parts)
        #distrib_plot(mfs_arr, elem, parts)      
        print("num_of_parts: ", np.array(mfs_arr).shape[0])
        ##scale and tsne
        t1_start = time.perf_counter()  
        stacked = np.hstack((mfs_arr, renie_arr))
        stacked = preprocessing.scale(stacked)
        #parts = pd.read_csv('parts.csv').values 
        #stacked = pd.read_csv('data/'+elem).values 
        #pd.DataFrame(stacked).to_csv(elem+'.csv', index=False)
        #stacked = reduce_features(stacked, 'pca')
        stacked = reduce_features(stacked, 'tsne', 2)
        #stacked = reduce_features(stacked, 'FA', 5)
        #distrib_plot(stacked, elem, parts)
        t2_start = time.perf_counter() 
        print("scale+tsne: ", int(t2_start - t1_start)//60, "min", (t2_start - t1_start) % 60, "sec")

        #clust
        t1_start = time.perf_counter() 
        k = find_best_k(stacked, 'gmm', elem, 'sil')
        clust = cluster(stacked, k, elem, parts, 'gmm', eps)
        val = assess(parts, clust)
        adj_scores.append(val)          
        with open('scores.txt', 'a+') as f:
            f.write(elem)
            f.write("  %s\n" % val)         
        t2_start = time.perf_counter() 
        print("clust: ", int(t2_start - t1_start)//60, "min", (t2_start - t1_start) % 60, "sec\n")

        #draw
        draw_clusters(stacked, clust, elem)
        draw_ground_truth(stacked, parts, elem)

    with open('scores.txt', 'a+') as f:
        f.write("Mean:")
        f.write("%s\n" % pd.DataFrame(adj_scores).mean())






