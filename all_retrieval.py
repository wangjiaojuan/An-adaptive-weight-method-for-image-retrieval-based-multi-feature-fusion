from __future__ import division 
from relation_function import*
import numpy as np
import os
import time



def compare(root):
    # get all_image label
    filepath = root+'label.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get feature
    CNN_feature = np.load(root+'binary_VGG_feature.npy')
    CNN1_feature = np.load(root+'binary_Alex_feature.npy')
    color_feature = np.load(root+'binary_color_feature.npy')
    ###################################################################################################################################
    # get retrieval image and database image
    path=0
    re=[] 
    database=[]
    while path<origion_line:
        re.append(path)
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start,end = name_map[re_image]
        num=end-start+1
        start=start+1
        while start<=end:
          database.append(start)
          start=start+1   
        path=path+num
    ############################################################################################################################
    ac1=0
    ac2=0
    ac3=0
    ac5=0
    ac6=0
    ac7=0
    #########################################################################################################################################
    r=0.4
    o=5
    #########################
    for i in xrange(len(re)):
        path=re[i]
        #print path
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start,end = name_map[re_image]
        ac_label=[]
        start=start+1
        while start<=end:
          ac_label.append(start)
          start=start+1 
        ################################### weight
        start_time = time.clock()
        CNN_distance=haming_distance(CNN_feature,path,database)
        result_CNN=query_result(CNN_distance,ac_label,database)
        color_distance=haming_distance(color_feature,path,database)
        result_color=query_result(color_distance,ac_label,database)
        CNN1_distance=haming_distance(CNN1_feature,path,database)
        result_CNN1=query_result(CNN1_distance,ac_label,database)
        ac1=ac1+result_CNN
        ac2=ac2+result_color
        ac3=ac3+result_CNN1
        #############################
        fix_ac=fix_query(CNN1_distance,CNN_distance,color_distance,ac_label,database)
        reback_ac=reback_query(result_CNN,result_color,result_CNN1,CNN_distance,color_distance,CNN1_distance,ac_label,database)
        ac5=ac5+fix_ac
        ac6=ac6+reback_ac
        #########################################
    ac1=ac1/len(re)
    ac2=ac2/len(re)
    ac3=ac3/len(re) 
    ac5=ac5/len(re)
    ac6=ac6/len(re)
    #########################################################################################################"
    print "........................................comparision is"
    print "CNN,color,CNN1,AVG,RF.................is",ac1,ac2,ac3,ac5,ac6
def with_Entropy(root):
    # get all_image label
    filepath = root+'label.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get feature
    CNN_feature = np.load(root+'binary_VGG_feature.npy')
    CNN1_feature = np.load(root+'binary_Alex_feature.npy')
    color_feature = np.load(root+'binary_color_feature.npy')
    ###################################################################################################################################
    # get retrieval image and database image
    path=0
    re=[] 
    database=[]
    while path<origion_line:
        re.append(path)
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start,end = name_map[re_image]
        num=end-start+1
        start=start+1
        while start<=end:
          database.append(start)
          start=start+1   
        path=path+num
    ############################################################################################################################
    ac1=0
    ac2=0
    ac3=0
    #########################################################################################################################################
    r=0.4
    o=5
    #########################
    for i in xrange(len(re)):
        path=re[i]
        print path
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start,end = name_map[re_image]
        ac_label=[]
        start=start+1
        while start<=end:
          ac_label.append(start)
          start=start+1 
        ################################### weight
        start_time = time.clock()
        CNN_distance=Entropy_haming_distance(CNN_feature,path,database)
        result_CNN=query_result(CNN_distance,ac_label,database)
        color_distance=Entropy_haming_distance(color_feature,path,database)
        result_color=query_result(color_distance,ac_label,database)
        CNN1_distance=Entropy_haming_distance(CNN1_feature,path,database)
        result_CNN1=query_result(CNN1_distance,ac_label,database)
        ac1=ac1+result_CNN
        ac2=ac2+result_color
        ac3=ac3+result_CNN1
        #########################################
    ac1=ac1/len(re)
    ac2=ac2/len(re)
    ac3=ac3/len(re)  
    ########################################################################################################"
    print "........................................with Entropy"
    print "CNN,color,CNN1.................is",ac1,ac2,ac3
def unsupervised(root):
    # get all_image label
    filepath = root+'label.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get feature
    CNN_feature = np.load(root+'binary_VGG_feature.npy')
    CNN1_feature = np.load(root+'binary_Alex_feature.npy')
    color_feature = np.load(root+'binary_color_feature.npy')
    ###################################################################################################################################
    # get retrieval image and database image
    path=0
    re=[] 
    database=[]
    while path<origion_line:
        re.append(path)
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start,end = name_map[re_image]
        num=end-start+1
        start=start+1
        while start<=end:
          database.append(start)
          start=start+1   
        path=path+num
    ############################################################################################################################
    ac7=0
    #########################################################################################################################################
    r=0.4
    o=5
    #########################
    for i in xrange(len(re)):
        path=re[i]
        print path
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start,end = name_map[re_image]
        ac_label=[]
        start=start+1
        while start<=end:
          ac_label.append(start)
          start=start+1 
        ################################### weight
        CNN_distance=Entropy_haming_distance(CNN_feature,path,database)
        result_CNN=query_result(CNN_distance,ac_label,database)
        color_distance=Entropy_haming_distance(color_feature,path,database)
        result_color=query_result(color_distance,ac_label,database)
        CNN1_distance=Entropy_haming_distance(CNN1_feature,path,database)
        result_CNN1=query_result(CNN1_distance,ac_label,database)
        ours_ac=unsupervised_fusion(CNN_distance,color_distance,CNN1_distance,ac_label,database,r,o)
        ac7=ac7+ours_ac
        print "ac.................is",result_CNN,result_color,result_CNN1
        #########################################
    ac7=ac7/len(re) 
    #########################################################################################################"
    print "........................................unsupervised"
    print "our.................!!!!!!!!!!!!!!!!!!!is",ac7 
def supervised(root):
    # get all_image label
    filepath = root+'label.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get feature
    CNN_feature = np.load(root+'binary_VGG_feature.npy')
    CNN1_feature = np.load(root+'binary_Alex_feature.npy')
    color_feature = np.load(root+'binary_color_feature.npy')
    ###################################################################################################################################
    # get retrieval image and database image
    path=0
    re=[] 
    database=[]
    while path<origion_line:
        re.append(path)
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start,end = name_map[re_image]
        num=end-start+1
        start=start+1
        while start<=end:
          database.append(start)
          start=start+1   
        path=path+num
    ############################################################################################################################
    ac7=0
    #########################################################################################################################################
    r=0.4
    o=5
    #########################
    for i in xrange(len(re)):
        path=re[i]
        print path
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('/')[0]
        start,end = name_map[re_image]
        ac_label=[]
        start=start+1
        while start<=end:
          ac_label.append(start)
          start=start+1 
        ################################### weight
        CNN_distance=Entropy_haming_distance(CNN_feature,path,database)
        result_CNN=query_result(CNN_distance,ac_label,database)
        color_distance=Entropy_haming_distance(color_feature,path,database)
        result_color=query_result(color_distance,ac_label,database)
        CNN1_distance=Entropy_haming_distance(CNN1_feature,path,database)
        result_CNN1=query_result(CNN1_distance,ac_label,database)
        ours_ac=ours_query(result_CNN,result_color,result_CNN1,CNN_distance,color_distance,CNN1_distance,ac_label,database,r,o)
        ac7=ac7+ours_ac
        #########################################
    ac7=ac7/len(re) 
    #########################################################################################################"
    print "........................................supervised"
    print "our.................!!!!!!!!!!!!!!!!!!!is",ac7    
#################################################
root='/share/home/math4/oldcaffe/wangjiaojuan/caffe-master/Apagerank/Holidays/'
compare(root)
#with_Entropy(root)
#supervised(root)
#unsupervised(root)
