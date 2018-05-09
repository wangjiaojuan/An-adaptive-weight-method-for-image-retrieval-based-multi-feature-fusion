from __future__ import division 
import numpy as np
import math

def ac_img_label(origion_file):
    origion_line = len(origion_file)
    name_map = {}
    for j in xrange(origion_line):
        image_name = origion_file[j]
        class_name = image_name.split('/')[0]
        if name_map.has_key(class_name):
           name_map[class_name][1]+=1
        else:
           name_map[class_name]=[j,j]
    return name_map
#################################################################################################################
def haming_distance(CNN_feature,path,database):
    re_feature=CNN_feature[path,:]
    distance=np.zeros(len(database))
    for j in xrange(len(database)):
       im_feature=CNN_feature[database[j],:]
       dist=np.sum(abs(re_feature-im_feature))
       distance[j]= dist
    distance=distance/(np.sum(distance))
    return distance
def feature_Entropy(feature):
    n=feature.shape[0]
    m=feature.shape[1]
    H_Entropy=[]
    for s in xrange(m):
        feature[:,s]=feature[:,s]/np.sum(feature[:,s])
    for s in xrange(m):
        HE=0
        for t in xrange(n):
            if feature[t,s]>0:
               HE=HE+feature[t,s]*math.log(feature[t,s],2)
        HHE=HE*((-1)/math.log(n,2))
        HHEE=math.exp(1-HHE)
        H_Entropy.append(HHE)
    H_Entropy=H_Entropy/np.sum(H_Entropy)
    return H_Entropy
def Entropy_haming_distance(CNN_feature,path,database):
    H_Entropy=feature_Entropy(CNN_feature)
    re_feature=CNN_feature[path,:]
    distance=np.zeros(len(database))
    for j in xrange(len(database)):
       im_feature=CNN_feature[database[j],:]
       dist=0
       A=abs(re_feature-im_feature)
       for yh in xrange(len(A)):
           if A[yh]!=0:
              dist=dist+H_Entropy[yh]
       distance[j]= dist
    distance=distance/(np.sum(distance))
    return distance
###############################################################################################################
def query_result(distance,ac_label,database):
    sub_distance = sorted(distance)
    listc=[]
    for t in xrange(len(sub_distance)):
        for h in xrange(len(distance)):
          if (sub_distance[t]==distance[h])and(not database[h] in listc):
              listc.append(database[h])
    ac=mean_average_precison(ac_label,listc)
    return ac
###########################################################################################################################
def mean_average_precison(ac_label,result):
    similar_num = 0
    mean_average_precison = 0
    for j in xrange(len(result)):
       if (result[j] in ac_label):
               similar_num += 1
               mean_average_precison += float(similar_num)/(j+1)
               #print mean_average_precison
    mean_average_precison = mean_average_precison/len(ac_label)
    return mean_average_precison
#################################################################################################
###############################################3
def fix_query(CNN1_distance,CNN_distance,color_distance,ac_label,database):
    w_CNN=1/3
    w_color=1/3
    w_CNN1=1/3
    distance=CNN1_distance*w_CNN1+CNN_distance*w_CNN+color_distance*w_color
    sub_distance = sorted(distance)
    listc=[]
    for t in xrange(len(sub_distance)):
        for h in xrange(len(distance)):
          if (sub_distance[t]==distance[h])and(not database[h] in listc):
              listc.append(database[h])
    ac=mean_average_precison(ac_label,listc)
    return ac
#######################################################################
def reback_query(result_CNN,result_color,result_CNN1,CNN_distance,color_distance,CNN1_distance,ac_label,database):
    global w_CNN,w_color,w_CNN1
    w_sum=result_CNN+result_color+result_CNN1
    if w_sum==0:
       w_CNN=1/3
       w_color=1/3
       w_CNN1=1/3
    else:
       w_CNN=result_CNN/w_sum
       w_color=result_color/w_sum
       w_CNN1=result_CNN1/w_sum
    distance=CNN1_distance*w_CNN1+CNN_distance*w_CNN+color_distance*w_color
    sub_distance = sorted(distance)
    listc=[]
    for t in xrange(len(sub_distance)):
        for h in xrange(len(distance)):
          if (sub_distance[t]==distance[h])and(not database[h] in listc):
              listc.append(database[h])
    ac=mean_average_precison(ac_label,listc)
    return ac
#############################
def ours_query(result_CNN,result_color,result_CNN1,CNN_distance,color_distance,CNN1_distance,ac_label,database,r,o):
    w_sum=result_CNN+result_color+result_CNN1
    if w_sum==0:
        B=[1/3,1/3,1/3];
    else:
        pre1=result_CNN
        pre2=result_color
        pre3=result_CNN1
        B=[1/3,1/3,1/3];
        A=[1/3,1/3,1/3];
        HH=[[1,pre2-pre1,pre3-pre1],[pre1-pre2,1,pre3-pre2],[pre1-pre3,pre2-pre3,1]];
        for ss in xrange(len(HH)):
            for tt in xrange(len(HH)):
                 if(HH[ss][tt]>0):
                    HH[ss][tt]=math.exp(o*HH[ss][tt]);
                 else:
                    HH[ss][tt]=abs(HH[ss][tt]);
        kkk=10000;
        while(kkk>0.0005):        
             A[0]=r*B[0]+(1-r)*(HH[0][0]*B[0]+HH[1][0]*B[1]+HH[2][0]*B[2])
             A[1]=r*B[1]+(1-r)*(HH[0][1]*B[0]+HH[1][1]*B[1]+HH[2][1]*B[2])
             A[2]=r*B[2]+(1-r)*(HH[0][2]*B[0]+HH[1][2]*B[1]+HH[2][2]*B[2])
             sss=A[0]+A[1]+A[2]
             A[0]=A[0]/sss
             A[1]=A[1]/sss
             A[2]=A[2]/sss
             B[0]=r*A[0]+(1-r)*(HH[0][0]*A[0]+HH[1][0]*A[1]+HH[2][0]*A[2])
             B[1]=r*A[1]+(1-r)*(HH[0][1]*A[0]+HH[1][1]*A[1]+HH[2][1]*A[2])
             B[2]=r*A[2]+(1-r)*(HH[0][2]*A[0]+HH[1][2]*A[1]+HH[2][2]*A[2])
             sss=B[0]+B[1]+B[2]
             B[0]=B[0]/sss
             B[1]=B[1]/sss
             B[2]=B[2]/sss
             kkk=np.sqrt((A[0]-B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1])+(A[2]-B[2])*(A[2]-B[2]))
    print "weight...............is",B[0],B[1],B[2]
    distance=CNN1_distance*B[2]+CNN_distance*B[0]+color_distance*B[1]
    sub_distance = sorted(distance)
    listc=[]
    #lo=0
    for t in xrange(len(sub_distance)):
        for h in xrange(len(distance)):
          if (sub_distance[t]==distance[h])and(not database[h] in listc):
              listc.append(database[h])
              #while (lo<4):
              #      print 1-distance[h]
              #      lo=lo+1
    #print listc[0:4]
    ac=mean_average_precison(ac_label,listc)
    return ac
###########################################################
def weight_Entropy(ddist):
    HE=0
    ddist=ddist/np.sum(ddist)
    #pdb.set_trace()
    for s in xrange(len(ddist)):
        if ddist[s]>0:
           HE=HE+ddist[s]*math.log(ddist[s],2)
    HHE=HE*((-1)/math.log(len(ddist),2))
    #HHEE=math.exp(1-HHE)
    return HHE
def supervised_fusion(distance1,distance2,distance3,ac_label,database,r,o):
    pre1=weight_Entropy(distance1)
    pre2=weight_Entropy(distance2)
    pre3=weight_Entropy(distance3)
    B=[1/3,1/3,1/3];
    A=[1/3,1/3,1/3];
    HH=[[1,pre2-pre1,pre3-pre1],[pre1-pre2,1,pre3-pre2],[pre1-pre3,pre2-pre3,1]];
    for ss in xrange(len(HH)):
        for tt in xrange(len(HH)):
             if(HH[ss][tt]>0):
                HH[ss][tt]=math.exp(o*HH[ss][tt]);
             else:
                HH[ss][tt]=abs(HH[ss][tt]);
    kkk=10000;
    while(kkk>0.0005):        
         A[0]=r*B[0]+(1-r)*(HH[0][0]*B[0]+HH[1][0]*B[1]+HH[2][0]*B[2])
         A[1]=r*B[1]+(1-r)*(HH[0][1]*B[0]+HH[1][1]*B[1]+HH[2][1]*B[2])
         A[2]=r*B[2]+(1-r)*(HH[0][2]*B[0]+HH[1][2]*B[1]+HH[2][2]*B[2])
         sss=A[0]+A[1]+A[2]
         A[0]=A[0]/sss
         A[1]=A[1]/sss
         A[2]=A[2]/sss
         B[0]=r*A[0]+(1-r)*(HH[0][0]*A[0]+HH[1][0]*A[1]+HH[2][0]*A[2])
         B[1]=r*A[1]+(1-r)*(HH[0][1]*A[0]+HH[1][1]*A[1]+HH[2][1]*A[2])
         B[2]=r*A[2]+(1-r)*(HH[0][2]*A[0]+HH[1][2]*A[1]+HH[2][2]*A[2])
         sss=B[0]+B[1]+B[2]
         B[0]=B[0]/sss
         B[1]=B[1]/sss
         B[2]=B[2]/sss
         kkk=np.sqrt((A[0]-B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1])+(A[2]-B[2])*(A[2]-B[2]))
    #print "weight...............is",B[0],B[1],B[2]
    distance=distance3*B[2]+distance1*B[0]+distance2*B[1]
    sub_distance = sorted(distance)
    listc=[]
    for t in xrange(len(sub_distance)):
        for h in xrange(len(distance)):
          if (sub_distance[t]==distance[h])and(not database[h] in listc):
              listc.append(database[h])
    ac=mean_average_precison(ac_label,listc)
    return ac