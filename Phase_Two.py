import numpy.random as npr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###
from scipy import stats
from scipy import special
from scipy import optimize
from scipy import integrate

###
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.metrics import explained_variance_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

###
from astropy.stats import jackknife_stats

###
import itertools
import sys


###TensorFlow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.engine import data_adapter
import tensorflow.keras.backend as K


####################################################
#### ADDITIONAL FUNCTIONS ##########################
####################################################

def check(x, list_):
    y = x.copy()
    for kk in range(y.shape[0]):
        if x[kk] in list_:
            y[kk] = True
        else:
            y[kk]= False
    return y
##
def theta(x, theta0= .5):
    
    return np.heaviside(x-theta0, 0) 

######################################################
######################################################
######################################################

class Mecatti_MFS:
    
    """
    Multi-Frame Survey Estimators (single frame multiplicity approach) for Ligurian EU-SILC data
    """
    
    def __init__(self, use_eusilc= True, n_sample= 1000):

        """
        Initial Function.
        use_eusilc --> Whether to load the EU-SILC data
        n_sample --> Sample size generated through the Gaussian Mixture Density Model. Valid only when "use_eusilc= False"
        """
      
        ###load data
        ### Take the Ligurian EU-SILC data and load it
        df = pd.read_csv('Liguria.csv')

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
        ##define equivalized household size (EHI)
        y1= df['CURRENT_E'].values ###Annual Income
        y2= df['MQ'].values ### Area of the dwelling 
        Y = np.vstack([y1, y2]).T
        
        ####evaluate equivalized household size
        eHS = 1+.5*(df['TOT_FATTO'].values-df['RAGA017'].values)+.3*df['RAGA017'].values
        eHI = (y1/eHS).reshape(-1, 1)

        ###MinMac scaling for EHI
        minmax = MinMaxScaler((0, 1))
        minmax.fit(eHI)
        self.Ystd = minmax.transform(eHI)

        ################################################################
        ###PREPARE THE MULTI-FRAME SCHEME

        ####Consider the ensemble of rare classes
        self.classes = ['CITTADX', 'ITALIA', 'RAGA017', 'RAGA115', 'SEV_MAT_DEPRIV'] 

        ### Extract the portion of data with the rare classes
        self.X = df[self.classes]
        
        ### Household with more than 3 children
        self.X['RAGA115'] = (self.X['RAGA115'].values>=3).astype(int)
        self.X['RAGA017'] = (self.X['RAGA017'].values>=3).astype(int)
        
        ### Subjects who were not born in Italy
        self.X['ITALIA'] =  (self.X['ITALIA'].values==2).astype(int)

        ### Subjects who had no italian citizenship
        self.X['CITTADX'] =  (self.X['CITTADX'].values==2).astype(int)
        
        ###Several Material Deprivation
        self.X['SEV_MAT_DEPRIV'] =  (self.X['SEV_MAT_DEPRIV'].values==1).astype(int)
        
        ####Determine the General Frame (i.e., all the others who do not fall into the rare classes)
        self.X['General_Frame'] = np.ones(self.X.shape[0])
        
        ###deteremine the total number of classes (the rare ones + the general one)
        self.number_classes = self.X.shape[1]

        ####add the explicative variable
        self.X['EHI'] =  self.Ystd

        #### list with all names of rare classes
        lista = ["General_Frame", 'CITTADX', 'ITALIA', 'RAGA017', 'RAGA115', 'SEV_MAT_DEPRIV']
        
        ###add multiplicity
        self.X['Multiplicity'] = 1/np.maximum(1, self.X[lista].sum(axis= 1).values)

        if use_eusilc == False:
            
            Xbin = self.X[['CITTADX', 'ITALIA', 'RAGA017', 'RAGA115', 'SEV_MAT_DEPRIV', 'EHI']].values
            
            ###FIT PCA
            ### fit PCA as an endomorphism to be sure that the whole variance is maintained
            pca = PCA(n_components=Xbin.shape[1])
            pca.fit(Xbin)
            
            #PCA latent description
            Xpca = pca.transform(Xbin)
            
            #USE Gaussian Mixture to fit the posterior of the Xpca; generate new samples
            GGMix = GenerateGaussianMixture(Xpca)
            optimal_nclue = GGMix.optimal_ncomponents(return_aic=False)
            print("Optimal Number of Gaussian:", optimal_nclue)
            
            ###get new sample
            Xpca_gen, __ = GGMix.generate(n_components=optimal_nclue, n_samples= n_sample)
            new_data = pca.inverse_transform(Xpca_gen)
            new_data[:, 0:5] = new_data[:, 0:5].round(0)
            
            ##make needed modifications
            self.X = pd.DataFrame(new_data, columns=['CITTADX', 'ITALIA', 'RAGA017', 'RAGA115', 'SEV_MAT_DEPRIV', 'EHI'])
            ##
            self.X['General_Frame'] = np.ones(self.X.shape[0])
            self.X['Multiplicity'] = 1/np.maximum(1, self.X[lista].sum(axis= 1).values)
          
        ### ### ###
        self.effective_pop = np.array([self.X[self.X[item]==1]['Multiplicity'].sum() for item in lista])
            
    def Mecatti_mean_estimator(self, df, proportions= None):

        """Unbiased Mean Estimator for MFS scheme with single frame multiplicity approach.
        
            df --> Sampling Frame. (Once the "Mecatti_MFS" is called, call and use the internal variable self.X)
            proportions -->  initial condition for the optimal proportions of each frame. Default values are unitary.
        """
        
        #####
        if all(proportions) == None:
            proportions = np.ones(self.number_classes)
            
        ##name of classes
        name_classes = np.hstack([['General_Frame'], self.classes])
        
        ##where all stuff will be saved
        rho = []
        
        ###for each class evaluate the estimator for the mean
        for kk in range(self.number_classes):
            
            ####
            prop =  proportions[kk]

            ### kk==0 --> General Frame
            if kk ==0: 
                
                ###make a simple random sampling
                yy = df.sample(frac=prop, random_state=29).index

                ###Evaluate the response variable over the multiplicity
                ehi_fold = np.average(yy['EHI'].values/prop, weights=yy['Multiplicity'].values)
                rho.append(ehi_fold)
              
            else:

                ### make a simple random sampling in a rare frame
                condition = df[name_classes[kk]] == 1

                ###Evaluate the response variable over the multiplicity
                yy = df[condition].sample(frac=prop, random_state=29)[['EHI', 'Multiplicity']]
                ehi_fold = np.average(yy['EHI'].values/prop, weights=yy['Multiplicity'].values)
                rho.append(ehi_fold)
              
        
        ##make a dictionary with response variables values (rescaled by the inverse of multiplicity)
        rho_dict = {name_classes[kk]:rho[kk] for kk in range(len(name_classes))}

        ###evalaute the estimator
        estimator_ = np.sum(rho)
        return estimator_
    
    
    def Mecatti_variance_estimator(self, proportions, df):
        
        """Unbiased Variance Estimator for MFS scheme with single frame multiplicity approach.
        
            df --> Sampling Frame. (Once the "Mecatti_MFS" is called, call and use the internal variable self.X)
            proportions -->  initial condition for the optimal proportions of each frame. Default values are unitary.
        """
        
      
        if all(proportions) == None:
            proportions = np.ones(self.number_classes)
            
        ##the name of rare classes
        name_classes = np.hstack([['General_Frame'], self.classes])
        
        ##where saving all stuff
        rho = []
        
        ###for each class evaluate Mecatti's estimator for the variance
        for kk in range(self.number_classes):
            
            ####
            prop = proportions[kk]

            ###General Frame
            if kk ==0:
                
                ###make a simple random sampling
                N = df.shape[0]
                yy = df.sample(frac=prop, random_state=29)[['EHI', 'Multiplicity']]
                n = yy.shape[0]
                
                ###various Terms
                amplitude = N*(1-prop)/((n**2)*(n-1))
                v0 = N*np.average(yy['EHI'].values**2, weights=(yy['Multiplicity'].values)**2)
                v1 = prop*(np.average(yy['EHI'].values, weights=yy['Multiplicity'].values)**2)
                
                ###determine response variable
                ehi_fold = amplitude*(v0+v1)

                ###append response varibale
                rho.append(ehi_fold)
            else:

                ###take one rare class
                condition = df[name_classes[kk]] == 1
                
                ###make a simple random sampling
                N = df[condition].shape[0]
                ###
                yy = df[condition].sample(frac=prop, random_state=29)[['EHI', 'Multiplicity']]
                n = N*yy.shape[0]
                
                ### various terms
                amplitude = N*(1-prop)/((n**2)*(n-1))
                v0 = N*np.average(yy['EHI'].values**2, weights=(yy['Multiplicity'].values)**2)
                v1 = prop*(np.average(yy['EHI'].values, weights=yy['Multiplicity'].values)**2)
                
                ### determine the response variable
                ehi_fold = amplitude*(v0+v1)

                ###append the value
                rho.append(ehi_fold)
                ### ###
                
        
        ##make a dictinary with EHI values
        rho_dict = {name_classes[kk]:rho[kk] for kk in range(len(name_classes))}

        ###determine the estimator
        estimator_ = np.sum(rho)
        
        return estimator_
    
    def Equivalized_SRS_varinace_estimator(self, proportions, df):

        """Readaption of SRS variance estimator for MFS scheme"""
      
        if all(proportions) == None:
            proportions = np.ones(self.number_classes)
            
        ##decide name of classes
        name_classes = np.hstack([['General_Frame'], self.classes])
        
        ##define the multiplicity of the elements
        #df['Multiplicity'] = self.Mecatti_multiplicity(df, proportions)
        
        ##where saving all stuff
        rho = []
        
        ###for each class take the estimator for the mean
        for kk in range(self.number_classes):
            
            ####
            prop = proportions[kk]
            
            if kk ==0:
                
                ###make a simple random sampling
                N = df.shape[0]
                yy = df.sample(frac=prop, random_state=29)[['EHI', 'Multiplicity']]
                neff = np.sum(yy['Multiplicity'].values)
                rho.append(neff)
            
            else:
                condition = df[name_classes[kk]] == 1
                ###make a simple random sampling
                N = df[condition].shape[0]
                ###
                yy = df[condition].sample(frac=prop, random_state=29)[['EHI', 'Multiplicity']]
                neff = np.sum(yy['Multiplicity'].values)
                rho.append(neff)
                ### ###
        
        #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        #### EVALUATE THE VARIANCE ESTIMATOR FOR AN SRS
        Nequivalent = np.sum(rho)
        N0= df.shape[0]
        prop_equiv = np.minimum(1, Nequivalent/N0)
        y_srs = df.sample(frac=prop, random_state=29)[['EHI']].values
        return np.var(y_srs)*(1-prop_equiv)*N**2
      
    def find_best_Vmfs_constrained(self, df, 
                                   pop_constraint= 100):

        """Solve the optimal problem to determine the proportion of samples to allocate to each incomplete frame of MFS scheme.
           df --> sampling frame.
           pop_constraint --> the total population involved in the sampling scheme.
        """
        
        ### name of classes
        name_classes = np.hstack([['General_Frame'], self.classes])
        self.name_classes_Vmfs = name_classes

        ### internal function. variance estimator
        def __internal__(X, df, v_ideal= 0):
            Vcmp = self.Mecatti_variance_estimator(X, df)
            return Vcmp   

        ### internal function. Constraint function.
        def __constraint__(X):
            constr = np.dot(X, self.effective_pop)-pop_constraint
            return constr

        ###determine the lower bond (less than 1/N with N the total size per each frame)                            
        lower_bound = np.min(1/(self.effective_pop+1))

        ####Solve the minimal problem                             
        res = optimize.minimize(__internal__, 
                                     x0= npr.uniform(lower_bound, 1, size= self.effective_pop.size),
                                     args= (df,),
                                 method='SLSQP', 
                                 constraints={'type': 'eq', 'fun': __constraint__}, 
                                 bounds=optimize.Bounds(lb = 1/self.effective_pop, ub = 1))                     
        return res.x
    
    
    def find_best_Vsrs_constrained(self, df, 
                                   pop_constraint= 100):

        """Solve the optimal problem to determine the proportion of samples to allocate to each incomplete frame of MFS scheme.
           df --> sampling frame.
           pop_constraint --> the total population involved in the sampling scheme.
        """

        #####
        name_classes = np.hstack([['General_Frame'], self.classes])
        self.name_classes_Vmfs = name_classes

        ###Interanl Function. SRS variance estimator                        
        def __internal__(X, df):
            Vcmp = self.Equivalized_SRS_varinace_estimator(X, df)
            return Vcmp   

        ### Constraint Function                             
        def __constraint__(X):
            constr = np.dot(X, self.effective_pop)-pop_constraint
            return constr

        ###lower bound per frame
        lower_bound = np.min(1/(self.effective_pop+1))

        ###Find the minimal solution
        res = optimize.minimize(__internal__, 
                                     x0= npr.uniform(lower_bound, 1, size= self.effective_pop.size),
                                     args= (df,),
                                 method='SLSQP', 
                                 constraints={'type': 'eq', 'fun': __constraint__}, 
                                 bounds=optimize.Bounds(lb = 1/self.effective_pop, ub = 1))
        
        return res.x
        
    
    


