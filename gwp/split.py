import csv
import numpy as np
import pandas as pd
import time
import random

def diversity_selecion(csvfile,ratio = 0.8,verbos = True):

    df = pd.read_csv(csvfile)
    n_samples = df.shape[0]
    N_features = df.shape[1] - 4
    
    data_file_name = csvfile

    diverse_ratio = ratio
    remaining_ratio = 1-diverse_ratio

    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        data = np.empty((n_samples, N_features))
        feature_names = np.array(temp)
        
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[2:2+N_features], dtype=np.float64)

    N_materials = data.shape[0]

    N_features = N_features
    
    feature_0 = data.T[0]
    feature_1 = data.T[1]
    feature_2 = data.T[2]
    feature_3 = data.T[3]
    feature_4 = data.T[4]
    feature_5 = data.T[5]
    feature_6 = data.T[6]
    feature_7 = data.T[7]
    feature_8 = data.T[8]
    feature_9 = data.T[9]
    
    feature_0 = (feature_0 - np.min(feature_0))/(np.max(feature_0) - np.min(feature_0))
    feature_1 = (feature_1 - np.min(feature_1))/(np.max(feature_1) - np.min(feature_1))
    feature_2 = (feature_2 - np.min(feature_2))/(np.max(feature_2) - np.min(feature_2))
    feature_3 = (feature_3 - np.min(feature_3))/(np.max(feature_3) - np.min(feature_3))
    feature_4 = (feature_4 - np.min(feature_4))/(np.max(feature_4) - np.min(feature_4))
    feature_5 = (feature_5 - np.min(feature_5))/(np.max(feature_5) - np.min(feature_5))
    feature_6 = (feature_6 - np.min(feature_6))/(np.max(feature_6) - np.min(feature_6))
    feature_7 = (feature_7 - np.min(feature_7))/(np.max(feature_7) - np.min(feature_7))
    feature_8 = (feature_8 - np.min(feature_8))/(np.max(feature_8) - np.min(feature_8))
    feature_9 = (feature_9 - np.min(feature_9))/(np.max(feature_9) - np.min(feature_9))

    if N_features == N_features:
        x = np.concatenate((feature_0.reshape(1,N_materials),feature_1.reshape(1,N_materials),feature_2.reshape(1,N_materials),
                            feature_3.reshape(1,N_materials),feature_4.reshape(1,N_materials),feature_5.reshape(1,N_materials),
                            feature_6.reshape(1,N_materials),feature_7.reshape(1,N_materials),feature_8.reshape(1,N_materials),
                            feature_9.reshape(1,N_materials)))
    N_sample = int(N_materials * diverse_ratio)-1
    

    time.sleep(1)
    # store indices of x here for the diverse and non-diverse sets.
    diverse_set = []
    remaining_set = list(range(N_materials))
    ### INITIALIZE WITH RANDOMLY SELECTED POINT
    idx_init = random.sample(list(np.arange(N_materials)),1)[0]
    diverse_set.append(idx_init)
    remaining_set.remove(idx_init)
    N_diverse = 1
    while N_diverse <= N_sample:
        print("Selecting point ", N_diverse)
        min_d_to_diverse_set = np.zeros((N_materials-N_diverse,))
        # for every candidate point not in diverse set...
        for i in range(N_materials - N_diverse):
            # get the distance of this point to each point in the diverse set
            d_from_each_diverse_pt = np.linalg.norm(x[:,diverse_set] - x[:,remaining_set[i]].reshape(N_features,1),axis=0)
            # get the closest distance that this point is to the diverse set
            min_d_to_diverse_set[i] = np.min(d_from_each_diverse_pt)
        # select point that has the largest distance from the diverse set
        idx_select = remaining_set[np.argmax(min_d_to_diverse_set)]
        assert (len(remaining_set) == np.size(min_d_to_diverse_set))
        print("\tSelected point " , idx_select)
        # add point to diverse set; remove it from remaining set
        diverse_set.append(idx_select)
        remaining_set.remove(idx_select)
        print("\tPts in diverse set: ", len(diverse_set))
        print("\tPts in remaining set: ", len(remaining_set))
        print(diverse_set[N_diverse-1])
        N_diverse += 1

    if verbos:
        print("Total number of materials : ", data.shape[0])
        print("Number of features: ", N_features)
        print("Shape of feature x: ", np.shape(x))
        print("Example feature vector = " , x[:,0])
        print("Sampling %d diverse structures out of %d" % (N_sample+1,N_materials))
        print("total accessible materials considered: ", N_materials)
        print("Starting diversity selection. Seeking %d points" % (N_sample+1))

    with open("divided_set_"+str(diverse_ratio)+"_"+str("%.1f"%remaining_ratio)+"_"+"data_split.txt", "w") as f:
        f.write(str(diverse_set)+" "+str(remaining_set))
        print("Save file name : divided_set_"+str(diverse_ratio)+"_"+str("%.1f"%remaining_ratio)+"_"+"data_split.txt")
