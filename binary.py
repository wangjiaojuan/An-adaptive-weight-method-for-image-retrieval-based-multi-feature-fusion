import numpy as np
#import matplotlib.pyplot as plt
import os





root='/share/home/math4/oldcaffe/wangjiaojuan/caffe-master/Apagerank/Holidays/'
VGG16_feature = np.load(root+"VGG_feature.npy")
n=VGG16_feature.shape[0]
m=VGG16_feature.shape[1]
kk=np.zeros((n,m)) 
for i in xrange(n):
    n_mean=np.sum(VGG16_feature[i,:])/m
    for j in xrange(m):
        if VGG16_feature[i,j]>=n_mean:
            kk[i,j]=1
        else:
            kk[i,j]=0
np.save(root+"binary_VGG_feature.npy",kk)
#############
VGG16_feature = np.load(root+"Alex_feature.npy")
n=VGG16_feature.shape[0]
m=VGG16_feature.shape[1]
kk=np.zeros((n,m)) 
for i in xrange(n):
    n_mean=np.sum(VGG16_feature[i,:])/m
    for j in xrange(m):
        if VGG16_feature[i,j]>=n_mean:
            kk[i,j]=1
        else:
            kk[i,j]=0
np.save(root+"binary_Alex_feature.npy",kk)
#############
VGG16_feature = np.load(root+"color_feature.npy")
n=VGG16_feature.shape[0]
m=VGG16_feature.shape[1]
kk=np.zeros((n,m)) 
for i in xrange(n):
    n_mean=np.sum(VGG16_feature[i,:])/m
    for j in xrange(m):
        if VGG16_feature[i,j]>=n_mean:
            kk[i,j]=1
        else:
            kk[i,j]=0
np.save(root+"binary_color_feature.npy",kk)
#############
VGG16_feature = np.load(root+"GIST_feature.npy")
n=VGG16_feature.shape[0]
m=VGG16_feature.shape[1]
kk=np.zeros((n,m)) 
for i in xrange(n):
    n_mean=np.sum(VGG16_feature[i,:])/m
    for j in xrange(m):
        if VGG16_feature[i,j]>=n_mean:
            kk[i,j]=1
        else:
            kk[i,j]=0
np.save(root+"binary_GIST_feature.npy",kk)