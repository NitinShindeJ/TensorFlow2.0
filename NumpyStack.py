## Image Plotting

import pandas as pd
import numpy as np
#%%
df = pd.read_csv("mnist_test.csv")
#%%
df.shape
#%%
M = df.as_matrix()
#%%
im = M[0, 1:]
im.shape
#%%
im = im.reshape(28,28)
im.shape
#%%
import matplotlib.pyplot as plt

plt.imshow(im)
#%%
M[0,0]
#%%
plt.imshow(im, cmap='gray')

plt.show()
#%%
from scipy.stats import norm
#%%
norm.pdf(0)
#%%
norm.pdf(0, loc=5, scale=10) ## probabilty of 0 when mean 5 & SD 10
#%%
#element wise cal
r = np.random.random(10)
norm.pdf(r)
#%%
#Log PDF
norm.logpdf(r)
# cumilative PDF
norm.cdf(r)
norm.logcdf(r)
#%%
r = np.random.randn(10000)
plt.hist(r, bins=100)
plt.show()
#%%
r = 10*np.random.randn(10000)+5
plt.hist(r, bins=100)
plt.show()
#%%
s = np.random.randn(10000, 2)
plt.scatter(s[:,0], s[:,1])
plt.show()
#%%
#Scaling second column
s[:,1]=5*s[:,1]+2
#%%
plt.scatter(s[:,0], s[:,1])
plt.axis('equal')
plt.show()
#%%
cov = np.array([[1,0.8],[0.8,3]])

from scipy.stats import multivariate_normal as mvn

mj = np.array([0,2])

t = mvn.rvs(mean=mj, cov=cov, size=1000)
#%%
plt.scatter(t[:,0],t[:,1])
plt.axis('equal')
plt.show()
#%%
t_np = np.random.multivariate_normal(mean=mj, cov=cov, size=1000)
plt.scatter(t_np[:,0],t_np[:,1])
plt.axis('equal')
plt.show()
#%%

 





































