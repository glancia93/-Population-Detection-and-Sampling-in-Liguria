import sys
import itertools
import traceback

###
import numpy as np
import numpy.random as npr
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.stats import jackknife_stats


from scipy import stats
from scipy import signal
from scipy import special
from scipy.linalg import orth, inv
from scipy.optimize import root, minimize, OptimizeResult

from datetime import datetime as dt

###
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score, explained_variance_score
from sklearn.metrics.pairwise import euclidean_distances, chi2_kernel, rbf_kernel
from sklearn.neighbors import KernelDensity, kneighbors_graph
from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
from sklearn.feature_selection import RFECV, f_regression, mutual_info_regression
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering

###tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.engine import data_adapter
import tensorflow.keras.backend as K
 
### EXTRAS
import EUSILC2019_LigData as lig


############################################################
class Univariate_Entropy_Selection():
    
    """An univeriate entropy-based feature selection"""
    
    def __init__(self, df, Df, name_variables):
        
        """Preparation steps
        *********************
            df --> dataset Liguria
            Df --> dataset Italy
            name_varibales --> categorcial varibales to use.
        """
        
        ###
        self.df = df
        self.name_vars = name_variables
        self.Df = Df
        
        return
    
    def get_dataset(self, return_df= False):
        
        """Get a NaNs free version of data"""
        
        ###get the whole dataset
        dataset = self.Df[self.name_vars].values
        
        ###consider nans; take only those with less than an overall 10% of missing data
        flag = np.mean(np.isnan(dataset), axis= 1)<= 10e-2
        
        knn_imp = KNNImputer()
        knn_imp.fit(dataset[flag])
        if return_df:
            return pd.DataFrame(knn_imp.transform(dataset[flag]), 
                                columns= self.name_vars[flag])
        else:
            return knn_imp.transform(dataset[flag])
    
    def number_classes_per_stratum(self):
        
        box= []
        count_err = 0
        for name in self.name_vars:
    
            ###consider only data with at least 100 occurences
            try:
                ###size
                prev_ = np.sum(~np.isnan(self.df[name].values))
                
                ##check whether we have enought size...
                if prev_ >= 100:

                    ###prepare data (remove nans)
                    data_2021 = self.df[name].values
                    data_2021 = data_2021[~np.isnan(data_2021)]
                    ######


                     ###make label encoding
                    LE_ = LabelEncoder()
                    LE_.fit(data_2021)
                    data_21_enc = LE_.fit_transform(data_2021)
                    nclasses_21 = LE_.classes_.size
                    Nbins = max(2, LE_.classes_.shape[0])
                    box.append([name, Nbins])
            except:
                count_err += 1
                
            
        print('COUNT_ERR:', count_err)

        return pd.DataFrame(np.vstack(box), columns=['Name_Var', 'Num_pop_class'])
        
        
    
    def eval_entropy(self):
        
        """evaluate entropy per each categorical variable"""
        
        box= []
        count_err = 0
        for name in self.name_vars:
    
        ###consider only data with at least 100 occurences
            try:
                ###size
                prev_ = np.sum(~np.isnan(self.df[name].values))
                
                ##check whether we have enought size...
                if prev_ >= 100:

                    ###prepare data (remove nans)
                    data_2021 = self.df[name].values
                    data_2021 = data_2021[~np.isnan(data_2021)]
                    ######

                    ###extremal values
                    #min_, max_ = min(data_2019.min(), data_2021.min()), max(data_2019.max(), data_2021.max())
                    ### Friedman Rule for Binning
                    #Size = data_2019.size + data_2021.size
                    #K = np.floor(np.log2(np.hstack([data_2019, data_2021]).size)+1)

                    ###make label encoding
                    LE_ = LabelEncoder()
                    LE_.fit(data_2021)
                    data_21_enc = LE_.fit_transform(data_2021)
                    nclasses_21 = LE_.classes_.size
                    Nbins = max(2, LE_.classes_.shape[0])

                    ###binning 
                    bins2021, __ = np.histogram(data_21_enc, bins=np.arange(Nbins+1))

                    #### #### #### ####
                    entropy = np.nan_to_num(stats.entropy(bins2021, base=Nbins), posinf=1)
                    ####
                    box.append([name, entropy])
            except Exception as e :
                print(e)
                count_err +=1
        
        return pd.DataFrame(box, columns=['Var_Name', 'Entropy'])
    
    
    def eval_collision_entropy(self):
        
        """evaluate entropy per each categorical variable"""
        
        box= []
        count_err = 0
        for name in self.name_vars:
    
        ###consider only data with at least 100 occurences
            try:
                ###size
                prev_ = np.sum(~np.isnan(self.df[name].values))
                
                ##check whether we have enought size...
                if prev_ >= 100:

                    ###prepare data (remove nans)
                    data_2021 = self.df[name].values
                    data_2021 = data_2021[~np.isnan(data_2021)]
                    ######

                    ###extremal values
                    #min_, max_ = min(data_2019.min(), data_2021.min()), max(data_2019.max(), data_2021.max())
                    ### Friedman Rule for Binning
                    #Size = data_2019.size + data_2021.size
                    #K = np.floor(np.log2(np.hstack([data_2019, data_2021]).size)+1)

                    ###make label encoding
                    LE_ = LabelEncoder()
                    LE_.fit(data_2021)
                    data_21_enc = LE_.fit_transform(data_2021)
                    nclasses_21 = LE_.classes_.size
                    Nbins = max(2, LE_.classes_.shape[0])

                    ###binning 
                    bins2021, __ = np.histogram(data_21_enc, bins=np.arange(Nbins+1))
                    bins2021 = bins2021/bins2021.sum()

                    #### #### #### ####
                    entropy = -np.log(np.sum((bins2021)**2))/np.log(Nbins)
                    ####
                    box.append([name, entropy])
            except Exception as e :
                print(e)
                count_err +=1
        
        return pd.DataFrame(box, columns=['Var_Name', 'Collision_Entropy'])
    
    def eval_Jensen_Shannon_divergence(self):
        
        """evaluate Jensen Shannon divergence per each categorical variable"""
        
        box= []
        count_err = 0
        for name in self.name_vars:
    
        ###consider only data with at least 100 occurences
            try:
                ###size
                prev_ = np.sum(~np.isnan(self.df[name].values))
                
                ##check whether we have enought size...
                if prev_ >= 100:

                    ###prepare data (remove nans)
                    data_2021 = self.df[name].values
                    data_2021 = data_2021[~np.isnan(data_2021)]
                    
                    ###national data
                    Data_2021 = self.Df[name].values
                    Data_2021 = Data_2021[~np.isnan(Data_2021)]

                    ###make label encoding -- liguria
                    LE_ = LabelEncoder()
                    LE_.fit(data_2021)
                    data_21_enc = LE_.fit_transform(data_2021)
                    Nbins_lig = max(2, LE_.classes_.shape[0])
                    
                    ###make label encoding -- Italia
                    LEN_ = LabelEncoder()
                    LEN_.fit(Data_2021)
                    Data_21_enc = LE_.fit_transform(Data_2021)
                    Nbins_it = max(2, LEN_.classes_.shape[0])
                    
                    ###determine Nbins
                    Nbins = max(Nbins_lig, Nbins_it)
                    
                    ###binning 
                    #liguria
                    bins2021, __ = np.histogram(data_21_enc, 
                                                bins=np.arange(Nbins+1), 
                                                weights= np.repeat(1/data_21_enc.size, data_21_enc.size))
                    #Italia
                    Bins2021, __ = np.histogram(Data_21_enc, 
                                                bins=np.arange(Nbins+1), 
                                                weights= np.repeat(1/Data_21_enc.size, Data_21_enc.size))
                    
                    ###define the "average distirbution"
                    M = .5*(bins2021+Bins2021)

                    #### #### #### ####
                    left_JS = np.nan_to_num(stats.entropy(bins2021, M, base=2), posinf=1)
                    right_JS = np.nan_to_num(stats.entropy(Bins2021, M, base=2), posinf=1)
                    
                    ####
                    box.append([name, .5*(left_JS+right_JS)])
                    
            except Exception as e :
                print(e)
                count_err +=1
        
        return pd.DataFrame(box, columns=['Var_Name', 'JS_Div'])
    
    
    def Shannon_similarity(self, X):
        
        """evaluate Jensen Shannon divergence per each categorical variable.
        
            X--> A matrix of data
        """
        
        combin_idx = list(itertools.combinations(range(X.shape[1]), 2))
        
        Similarity= np.zeros((X.shape[1], X.shape[1]))
        np.fill_diagonal(Similarity, val=1)
        
        box= []
        count_err = 0
        for item in combin_idx:
        
            #get the indeces
            ii, jj = item
                        
        ###consider only data with at least 100 occurences
            try:
                ###size
                prev_ii = np.sum(~np.isnan(X[:, ii]))
                prev_jj = np.sum(~np.isnan(X[:, jj]))
                prev_ = np.minimum(prev_ii, prev_jj)
                                
                ##check whether we have enought size...
                if prev_ >= 100:

                    ###Data_ii
                    data_ii = X[:, ii]
                    
                    ###Data_jj
                    data_jj = X[:, jj]
                    
                    ###make label encoding -- data_ii
                    LE_ = LabelEncoder()
                    LE_.fit(data_ii)
                    data_ii_enc = LE_.fit_transform(data_ii)
                    Nbins_ii = max(2, LE_.classes_.shape[0])
                    
                    ###make label encoding -- data_jj
                    LE_ = LabelEncoder()
                    LE_.fit(data_jj)
                    data_jj_enc = LE_.fit_transform(data_jj)
                    Nbins_jj = max(2, LE_.classes_.shape[0])
                    
                    
                    ###Binning
                    Hist, _x, _y,  = np.histogram2d(data_ii, data_jj, 
                                                bins=[np.arange(Nbins_ii+1), np.arange(Nbins_jj+1)])
                    
                    Hist = Hist/Hist.sum()
                                        
                    ##evaluate simialrity Metric
                    metric = -np.sum(special.xlogy(Hist, Hist)/np.log(Hist.size))
                    
                    ####
                    Similarity[ii, jj] = metric
                    Similarity[jj, ii] = metric                    
                    
            except Exception as e :
                print(e)
                Similarity[ii, jj] = -1
                Similarity[jj, ii] = -1
        
        return Similarity
    
    
    
    def weighted_euclidean_distance(self, X, Y, alpha= .5):
        
        Z = np.zeros((X.shape[0], Y.shape[0]))
                
        for ii in range(X.shape[0]):
            for jj in range(Y.shape[0]):
                    
                dist_ = np.sqrt(alpha*(X[ii, 0]-Y[jj, 0])**2 + (1-alpha)*(X[ii, 1]-Y[jj, 1])**2)
                dist_ *= np.sqrt(2)
                Z[ii, jj] = dist_
                
        return Z
                    
                    
    def get_entropy_features(self):
        
        d1 = self.eval_entropy()
        d1['compl_Entropy'] = 1-d1['Entropy'].values
        d2 = self.eval_Jensen_Shannon_divergence()
        d2['compl_JS_Div'] = 1 - d2['JS_Div']
        d3 = self.eval_collision_entropy()
        d3['compl_Collision_Entropy'] = 1 - d3['Collision_Entropy']
        
        
        DF = reduce(lambda left, right:     
                    pd.merge(left, right,
                    on = ['Var_Name'],
                    how = "outer"),
                    [d1, d2, d3])

        
        ###class attribution
        #coordinates_ = DF[['compl_Entropy', 'JS_Div']].values
        #fixed_centroids = np.array([[0, 0], [1, 1]]) 
        #class_ = self.weighted_euclidean_distance(coordinates_, fixed_centroids, alpha=.75).argmin(axis= 1)
        ###
        #DF['Class'] = class_
        
        ###condition (remove all features with 0 entropy)
        #condition = DF['Entropy'].values != 0
        
        return DF 

###################################################################

class Univariate_MomentsBased_selection():

    """A Class for estimating the Moment-based indexes"""
    
    def __init__(self, df, Df, name_variables, max_moment= 4):
        
        """
        Preparation steps
         *********************
        df --> dataset Liguria
        Df --> dataset Italy
        name_varibales --> categorcial varibales to use.
        """
            
        ###
        self.df = df
        self.name_vars = name_catgs
        self.Df = Df
        self.max_moment = max_moment
        return
    
    def get_dataset(self):
        
        ###get the whole dataset
        dataset = self.Df[self.name_vars].values
        
        ###consider nans; take only those with less than an overall 10% of missing data
        flag = np.mean(np.isnan(dataset), axis= 1)<= 10e-2
        
        knn_imp = KNNImputer()
        knn_imp.fit(dataset[flag])
        return knn_imp.transform(dataset[flag])
    
    def number_classes_per_stratum(self):
        
        box= []
        count_err = 0
        for name in self.name_vars:
    
            ###consider only data with at least 100 occurences
            try:
                ###size
                prev_ = np.sum(~np.isnan(self.df[name].values))
                
                ##check whether we have enought size...
                if prev_ >= 100:

                    ###prepare data (remove nans)
                    data_2021 = self.df[name].values
                    data_2021 = data_2021[~np.isnan(data_2021)]
                    ######


                     ###make label encoding
                    LE_ = LabelEncoder()
                    LE_.fit(data_2021)
                    data_21_enc = LE_.fit_transform(data_2021)
                    nclasses_21 = LE_.classes_.size
                    Nbins = max(2, LE_.classes_.shape[0])
                    box.append([name, Nbins])
            except:
                count_err += 1
                
            
        print('COUNT_ERR:', count_err)

        return pd.DataFrame(np.vstack(box), columns=['Name_Var', 'Num_pop_class'])
        
        
    def ReducedMomentFeats(self, min_moment= 3, max_moment= 5):
        
        Output = []   
        count_err = 0
        
        for name in self.name_vars:
            
            box = []
            
            ###consider only data with at least 100 occurences
            try:
                ###size
                prev_ = np.sum(~np.isnan(self.df[name].values))
                
                ##check whether we have enought size...
                if prev_ >= 100:

                    ###prepare data (remove nans)
                    data_2021 = self.df[name].values
                    data_2021 = data_2021[~np.isnan(data_2021)]
                    ######

                    ###make label encoding
                    LE_ = LabelEncoder()
                    LE_.fit(data_2021)
                    data_21_enc = LE_.fit_transform(data_2021)
                    nclasses_21 = LE_.classes_.size
                    Nbins = max(2, LE_.classes_.shape[0])

                    ###binning 
                    bins2021, __ = np.histogram(data_21_enc, bins=np.arange(Nbins+1))
                    density2021 = bins2021/bins2021.sum()
                    
                    ###this is the normalization with respsct to the distributions
                    gini_pure = np.sqrt(np.dot(density2021, density2021))
                    
                    #### #### #### ####
                    for jj in range(min_moment, max_moment):
                        
                        if jj ==1:
                            #moment = data_21_enc.mean()
                            #powers_ = np.power(np.arange(Nbins), jj)
                            ####this is the dot product
                            #dot_moment = np.dot(density2021, powers_)
                            ###this is the normalization with respsct to the powers
                            #sum_of_pows = np.sqrt(np.dot(powers_, powers_))
                            ####save in box
                            box.append(gini_pure)
                        
                        else:
                            moment = stats.moment(data_21_enc.mean(), moment=jj)
                            powers_ = np.power((np.arange(Nbins)-data_21_enc.mean()), jj)
                            ###this is the dot product
                            dot_moment = np.dot(density2021, powers_)
                            ###this is the normalization with respsct to the powers
                            sum_of_pows = np.sqrt(np.dot(powers_, powers_))
                            ####save in box
                            if jj%2==0:
                                box.append(np.abs(dot_moment/(sum_of_pows*gini_pure)))
                            else:
                                 box.append(1-np.abs(dot_moment/(sum_of_pows*gini_pure)))
                        
                        
                    ####save in Output
                    Output.append(box)
            except Exception as e :
                print(e)
                count_err +=1
        
        Outupt_array = np.vstack(Output)
        return Outupt_array


#################################
#################################
#################################

class AutoEncoder:

  """Multi-Variate approach for identifing rare items"""
  
    def __init__(self):
    
        return
    
    def AE(self, Xdata, 
                  units= 8, 
                  bottleneck= 8,
                  DropOut= 10e-2,
                  activation = 'tanh',
                  deepness= 3, 
                  lr = 1e-3, 
                  verbose= 1, 
                  epochs= 100,
                  batch_size = 32):
        
        """Variational Auto Encoder

        Xdata --> the input data;
        units --> units in each Dense layer
        bottleneck --> units in the bottleneck
        DropOut --> one-mean gaussin noise magnitude 
        activation --> activation function,
        deepness --> number of consecutive layers along both encoder and decoder 
        lr --> learning rate of the optmizer, 
        verbose --> verbose modality during the fit
        epochs --> maximal number of Epochs during the training phase
        batch_size --> batch size during the training phase
        """
        

        #### ### ### #### 
        ### AUTOENCORDER
        #### ### ### ####
    
        
        ## Input
        dim = Xdata.shape[1]
        Input = tf.keras.Input(Xdata.shape[1])
        X = tf.keras.layers.Lambda(lambda x : x)(Input)  

        
        for kk in range(deepness):
        
            X = tf.keras.layers.Dense(units= units,
                                    activation=None, 
                                    use_bias = False)(X)
            X = tf.keras.layers.Activation(activation)(X)
            X = tf.keras.layers.GaussianDropout(rate= DropOut)(X)   

        ###### ###### ######     
        ### final encoding -- VAE
       
        z_mean = tf.keras.layers.Dense(bottleneck, 
                                           name="z_mean", 
                                           activation= 'linear', 
                                           use_bias = False)(X)
        z_log_var = tf.keras.layers.Dense(bottleneck, 
                                           name="z_var", 
                                           activation= 'linear', 
                                           use_bias = False)(X)
        ####
        encoding = NormalSampling()([z_mean, z_log_var])
        
        
        ### ### ### ### ###
        # DEFINE ENCODER
        Encoder = tf.keras.models.Model(inputs= Input, outputs= encoding)
        
        
        #### #### ####
        ## DECODER ###
        ### Latent input
        latent_input = tf.keras.Input(shape= (encoding.shape[1]))
        Y = tf.keras.layers.Lambda(lambda x: x)(latent_input)
        
        ### other layers
        for kk in range(deepness-1):
            Y = tf.keras.layers.Dense(units= units, 
                                      activation=None, 
                                      use_bias = False)(Y)
            Y = tf.keras.layers.Activation(activation)(Y)
            Y = tf.keras.layers.GaussianDropout(rate= DropOut)(Y) 

        ###last decoding layer
        Y = tf.keras.layers.Dense(units= dim, 
                                      activation=None, 
                                      use_bias = False)(Y)
        
        Y = tf.keras.layers.Activation(activation)(Y)
        decoding = tf.keras.layers.GaussianDropout(rate= DropOut)(Y)  
        
        ### Define Decoder    
        Decoder = tf.keras.models.Model(inputs= latent_input, outputs= decoding)

        ####################
        ### AutoEncoder#####
        ae_bottleneck = Encoder(Input)
        ae_output = Decoder(ae_bottleneck)
        mymodel = tf.keras.models.Model(inputs= Input, outputs= ae_output)

        ###
        adam = tf.keras.optimizers.Adam(lr= lr)
        mse = 'mean_squared_error'
        bce = tf.keras.losses.BinaryCrossentropy()
        mymodel.compile(optimizer= adam, loss = mse)

        ### Fit
        Earlystop = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=10)
        history = mymodel.fit(Xdata, Xdata,
                               verbose= verbose, 
                               epochs= epochs, 
                               batch_size= batch_size)

        return mymodel, Encoder, Decoder


class AnomalyDetection:

   """A Python class to determine Anomalous items from a fitted AE"""
  
    def __init__(self, ae):

        """ae--> a Fitted keras AutoEncoder"""
      
        self.ae = ae
        
    def RecError(self, X):

        """Determine the Reconstruction Error
          X --> input data
        """
      
        RE = np.abs(self.ae.predict(X, verbose= 0)-X)
        return RE
    
    def AnomalyThreshold(self, X):
        """Determine vectorized threshold for unsupervised anomaly detection
          X --> input data
        """
        re = self.RecError(X)
        return re.max(axis= 0)
    
    def AnomalyFlag(self, Xtest, Xtrain):
        
        """Flag anomalous items.
          Xtrain --> Data involed in the training phase of the AE
          Xtest --> Data utilized to validate the AE
        """
      
        ##evalaute the threshold
        TH = self.AnomalyThreshold(Xtrain)
        RE_TEST = self.RecError(Xtest)
        
        ##
        FLAGS = [bool((item-TH>0).sum()) for item in RE_TEST]
        return np.array(FLAGS)
        
    def make_feature_permutation(self, X, feat_pos):
        
        """Function to make a feature permutation with the sscope of determining unrepresented classes"""
        
        Y = X.copy()
        Y[:, feat_pos] = npr.choice(Y[:, feat_pos], replace= False, size= Y.shape[0])
        return Y
    
    def PemutationImportance(self, Xtest, Xtrain, n_permutations= 30):
        
        """Importanced based on Anomaly"""

        ###empty boxes
        OUTPUT_mean = []
        OUTPUT_std = []
        OUTPUT_CI = []
        
        I0 = self.AnomalyFlag(Xtest, Xtrain).mean()
        
        ###make a cycle over the number of features
        for nfeat in range(Xtest.shape[1]):
            
            Importance= []
            ###perform per each pemrutation
            for nn in range(n_permutations):
                
                ###make pemrutation and evaluate the change in Importance (Anomaly)
                X_perm = self.make_feature_permutation(Xtest, nfeat)
                ImportanceFlags = self.AnomalyFlag(X_perm, Xtrain).mean()
                Importance.append(ImportanceFlags.mean())
                
            ###jackknife resampling for making estimation...
            jack_importance= jackknife_stats(np.array(Importance), np.mean)
        
            ###Lists where saving the outputs
            OUTPUT_mean.append(jack_importance[0])
            OUTPUT_std.append(np.sqrt(jack_importance[2]))
            OUTPUT_CI.append(jack_importance[3])
        
        ###Output as a Dictonoary
        Output = {'Importance_mean': np.array(OUTPUT_mean),
                  'Importance_change_mean': .5*(np.array(OUTPUT_mean)-I0),
                 'Importance_std': np.array(OUTPUT_std),
                 'Importance_CI': np.array(OUTPUT_CI)}
        return Output
