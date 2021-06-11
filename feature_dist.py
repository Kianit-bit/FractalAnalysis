import statsmodels.api as sm
from scipy.stats import norm
import pylab
import seaborn as sns
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

def distrib_plot(data, name, parts):
  d = pd.DataFrame(data)
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

  first = d.iloc[first]
  second = d.iloc[second]
  third = d.iloc[third]
  fourth = d.iloc[fourth]

  name = name[:-4]

  for i in range(0, len(d.columns)):
    plt.figure(figsize=(10,10))
    plt.hist(first.iloc[:,i], 100, density=True, histtype='step', stacked=True, fill=False)
    plt.hist(second.iloc[:,i], 100, density=True, histtype='step', stacked=True, fill=False)
    plt.hist(third.iloc[:,i], 100, density=True, histtype='step', stacked=True, fill=False)
    plt.hist(fourth.iloc[:,i], 100, density=True, histtype='step', stacked=True, fill=False)
    plt.legend(('1', '2', '3', '4'), loc='upper right')
    plt.savefig("graph/"+name+"_"+ str(i) +"_hist.png", dpi=300)
    #plt.show()
    plt.close()

    sm.qqplot(first.iloc[:,i], line='45', dist="t", fit=True) #'expon', 'norm'
    plt.savefig("graph/"+name+ "_" + str(i) +"_qq1.png", dpi=300)
    plt.close()

    sm.qqplot(second.iloc[:,i], line='45', dist="t", fit=True)
    plt.savefig("graph/"+name+ "_" + str(i) +"_qq2.png", dpi=300)
    plt.close()

    sm.qqplot(third.iloc[:,i], line='45', dist="t", fit=True)
    plt.savefig("graph/"+name+ "_" +str(i) +"_qq3.png", dpi=300)
    plt.close()

    sm.qqplot(fourth.iloc[:,i], line='45', dist="t", fit=True)
    plt.savefig("graph/"+name+ "_" +str(i) +"_qq4.png", dpi=300)
    plt.close()

  p_vals = []
  for i in range(0, len(first.columns)//2):
    for m in range(0, len(first.index), 100):
        p_vals.append(shapiro(first.iloc[m:m+100,i])[1])
        p_vals.append(shapiro(second.iloc[m:m+100,i])[1])
        p_vals.append(shapiro(third.iloc[m:m+100,i])[1])
        p_vals.append(shapiro(fourth.iloc[m:m+100,i])[1])

  plt.hist(p_vals[::4], bins=20)
  plt.show()
  plt.hist(p_vals[1::4], bins=20)
  plt.show()
  plt.hist(p_vals[2::4], bins=20)
  plt.show()
  plt.hist(p_vals[3::4], bins=20)
  plt.show()
  
  print(len([x for x in p_vals[::4] if x < 0.05])/len(p_vals[::4]))
  print(len([x for x in p_vals[1::4] if x < 0.05])/len(p_vals[1::4]))
  print(len([x for x in p_vals[2::4] if x < 0.05])/len(p_vals[2::4]))
  print(len([x for x in p_vals[3::4] if x < 0.05])/len(p_vals[3::4]))

  #Используется корреляция Пирсона?
  df_corr = d.corr()
  #df_corr.to_csv("graph/"+name+"corr.csv")
  fig, ax = plt.subplots(figsize=(10, 8))
  # mask
  mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
  # adjust mask and df
  mask = mask[1:, :-1]
  corr = df_corr.iloc[1:,:-1].copy()
  # plot heatmap
  sns.set(font_scale=0.4)
  sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='Blues',\
            vmin=-1, vmax=1, cbar_kws={"shrink": .8})
  # yticks
  plt.yticks(rotation=0)
  plt.savefig("graph/"+name+"_train_corrs.png", dpi=3000)
  #d.describe().to_csv("graph/"+name+"my_description.csv")
  plt.close()
