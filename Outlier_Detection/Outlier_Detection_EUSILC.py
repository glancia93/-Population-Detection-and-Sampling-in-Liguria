import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, describe, chi2_contingency, chisquare
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import matthews_corrcoef, accuracy_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import LeaveOneOut, KFold, LeavePOut
from sklearn.decomposition import PCA, KernelPCA
from astropy.stats import jackknife_stats
import seaborn as sns

###tensorflow
import tensorflow as tf
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import keras.backend as K



#Momenanly, put here all the classes and functions which will be useful later

class Utils:

    @staticmethod
    def bin_ages(age_series):
        """A simple function that groups ages into bins."""
        
        bins = [0, 16, 32, 64, np.inf]
        labels = [
            1,
            2,
            3,
            4
        ]
        return pd.cut(age_series, bins=bins, labels=labels, right=False)

    @staticmethod
    def categorization_dict():
        mydict = {
        "ETAINT" : {0:"Not applicable", 1:"0-16", 2:"16-32", 3:"32-64", 4:"Over 64"},
        'CITTADX': {0:"Not applicable", 1:"YES", 2:"NO"},
        'NCITT': {0:"Not applicable", 1:"YES", 2:"NO"},
        'SECITT': {0:"Not applicable", 1:"YES", 2:"NO"},
        'ITA': {0:"Not applicable", 1:"YES", 2:"NO"},
        'VIFAM': {0:"Not applicable", 1:"YES", 2:"NO, only for a period", 3:"Joined the household in 2019"},
        'LAVPRI':  {0:"Not applicable", 1:"Employed", 2:"Unemployed", 3:"Retired", 4:"Other"},
        'FONTERED':  {0 : "Not applicable",  
                      1 : "Income from salaried employment",  
                      2 : "Income from self-employment",  
                      3 : "Pensions",  
                      4 : "Benefits or other subsidies", 
                      5 : "Income from renting",
                      6: "Securities, stocks, and other investments",
                      7: "Support from cohabiting family members",
                      8: "Support from non-cohabiting family members"},
        'TIPSCU': { 
        0: "Not applicable",
        1: "Upper secondary school",
        2: "Lower secondary school",
        3: "Primary school",
        4: "Kindergarten",
        5: "Nursery",
        6: "No schooling"},
        'STACIV': {
        0: "Not applicable",
        1: "Single",
        2: "Married (living with spouse)",
        3: "Married (not living with spouse – de facto separated)",
        4: "Legally separated",
        5: "Divorced",
        6: "Widowed",
        7: "In a civil union",
        8: "Previously in a civil union (union ended)",
        9: "Previously in a civil union (union dissolved)",
        10: "Previously in a civil union (partner deceased)"},
        'TF': {
        0: "Not applicable",
        1: "Single under 35 years old",
        2: "Single aged 35–64",
        3: "Single aged 65 and over",
        4: "Couple without children – Woman under 35",
        5: "Couple without children – Woman aged 35–64",
        6: "Couple without children – Woman aged 65 and over",
        7: "Couple with at least one minor child",
        8: "Couple with only adult children",
        9: "Single parent with at least one minor child",
        10: "Single parent with only adult children",
        11: "Two or more family units",
        12: "Other family type"},
        'SEV_MAT_DEPRIV': {0:"Not applicable", 1:"YES", 2:"NO"},
        'RISKPOV': {0:"Not applicable", 1:"YES", 2:"NO"},
        'LOW_WORK_INT': {0:"NO", 1:"YES", 2:"Not Applicable"},
        'POV_SOC_EXCL': {0:"NO", 1:"YES", 2:"Not Applicable"},
        'QUINTI_EU': {0:"Not Applicable", 1:"1st", 2:"2nd", 3:"3rd", 4:"4th", 5:"5th"}}
        return mydict

    @staticmethod
    def features_dict():
        mydict = {
        "ETAINT" : "Respondent's age",
        'CITTADX': 'Italian Cizitenship',
        'NCITT': 'Italian citizen from birth',
        'SECITT': 'Secondary Cizitenship (if applicable)',
        'ITA': 'Nerver moved out of Italy',
        'VIFAM': 'Lived within the household for the whole 2019',
        'LAVPRI': 'Employment status or occupation type',
        'FONTERED': 'Source of income or funding',
        'TOT_TUTTI': 'Number of individuals in the household',
        'RAGA017': 'Number of minor Children within the household',
        'DON1555': 'Number of women aged 15-55',
        'STACIV': 'Civil status',
        'TF': 'Type of family',
        "TIPSCU" : "Type of school attended (if applicable)",
        'SEV_MAT_DEPRIV': 'Severity of material deprivation',
        'RISKPOV': 'Risk of poverty indicator',
        'LOW_WORK_INT': 'Indicator of low work intensity',
        'POV_SOC_EXCL': 'Social exclusion due to poverty',
        'QUINTI_EU': 'Quintile in the European Union economic ranking'
    }
        return mydict


    @staticmethod
    def cramers_v_weighted(x, y, weights):
        """
        Compute Cramér's V between two categorical vectors, weighted by sample weights.
        
        Parameters:
        - x, y: 1D array-like, categorical variables
        - weights: 1D array-like of sample weights
        
        Returns:
        - Cramér's V (float)
        """
        df = pd.DataFrame({"x": x, "y": y, "w": weights})
        contingency = pd.pivot_table(df, index="x", columns="y", values="w", aggfunc='sum', fill_value=0)
        
        chi2, _, _, _ = chi2_contingency(contingency, correction=False)
        n = contingency.values.sum()
        r, k = contingency.shape
        return np.sqrt(chi2 / (n * (min(k, r) - 1))) if min(k, r) > 1 else 0.0


    @staticmethod
    def cramers_v_matrix_weighted(X, weights):
        """
        Compute pairwise Cramér's V matrix for a 2D categorical matrix, weighted by sample weights.
    
        Parameters:
        - X: array-like or DataFrame, shape (n_samples, n_features)
        - weights: array-like, shape (n_samples,)
    
        Returns:
        - np.ndarray of shape (n_features, n_features) with pairwise Cramér’s V
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
    
        n_features = X.shape[1]
        V = np.zeros((n_features, n_features))
    
        for i in range(n_features):
            for j in range(i, n_features):
                v = cramers_v_weighted(X.iloc[:, i], X.iloc[:, j], weights)
                V[i, j] = v
                V[j, i] = v
    
        return V

    @staticmethod
    def make_psd(matrix, threshold=1e-8):
        """
        Project a symmetric matrix to the nearest PSD matrix.
        """
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < threshold] = 0.0  # remove small/negative values
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


    @staticmethod
    def map_numeric_to_string(df, mappings):
        """
        Converts numeric categories to human-readable strings based on feature-specific mappings.
    
        Parameters:
        - df: pd.DataFrame with columns ['Feature', 'Category']
        - mappings: dict of dicts specifying how to map each feature's numeric values to strings
    
        Returns:
        - pd.DataFrame with mapped 'Category' values
        """
        df = df.copy()
        df.columns = ["Feature", "Category"]  # enforce naming
        df["Category"] = df.apply(
            lambda row: mappings.get(row["Feature"], {}).get(row["Category"], f"{row['Feature']}={row['Category']}"),
            axis=1
        )
        return df

    @staticmethod
    def map_long_format_categories(df, mappings):
        """
        Maps numeric category codes to string labels in a long-format DataFrame.
    
        Parameters:
        - df: pd.DataFrame where the first column is 'Feature' and the others are category values
        - mappings: dict of dicts, where mappings[feature][code] gives the string for that code
    
        Returns:
        - pd.DataFrame with mapped category values
        """
        df = df.copy()
        feature_col = df.columns[0]
        value_cols = df.columns[1:]
    
        # Apply mapping row-wise
        for col in value_cols:
            df[col] = df.apply(
                lambda row: mappings.get(row[feature_col], {}).get(row[col], f"{row[feature_col]}={row[col]}"),
                axis=1
            )
    
        return df
    
    @staticmethod
    def jackknife_bias_corrected_xy(x, y, T):
        """
        Jackknife bias-corrected estimate and standard error for a statistic T(x, y).
        
        Parameters:
        - x: array-like, shape (n, ...)
        - y: array-like, shape (n, ...)
        - T: callable, function of (x, y) returning a scalar statistic
        
        Returns:
        - T_jack: bias-corrected jackknife estimate
        - T_mean: mean of the leave-one-out statistics
        - SE: standard error
        """
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        
        T_i = np.zeros(n)
        
        for i in range(n):
            x_loo = np.delete(x, i, axis=0)
            y_loo = np.delete(y, i, axis=0)
            T_i[i] = T(x_loo, y_loo)
        
        T_mean = T_i.mean()
        T_full = T(x, y)
        
        T_jack = n * T_full - (n - 1) * T_mean
        SE = np.sqrt((n - 1) * np.mean((T_i - T_mean)**2))
        
        return T_jack, T_mean, SE


    @staticmethod
    def compute_modes(df, inlier_mask, outlier_mask):
        """
        Compute the mode for each categorical feature separately for inliers and outliers.
    
        Parameters:
        - df: pd.DataFrame with categorical features
        - inlier_mask: boolean mask or index list for inliers
        - outlier_mask: boolean mask or index list for outliers
    
        Returns:
        - pd.DataFrame with columns: ['Feature', 'Mode_Inliers', 'Mode_Outliers']
        """
        inliers = df.loc[inlier_mask]
        outliers = df.loc[outlier_mask]
    
        modes_inliers = inliers.mode().iloc[0]  # first mode per feature
        modes_outliers = outliers.mode().iloc[0]
    
        summary_df = pd.DataFrame({
            'Feature': df.columns,
            'Mode_Inliers': modes_inliers.values,
            'Mode_Outliers': modes_outliers.values
        })
    
        return summary_df



class EntropyScore:
    def __init__(self, data, weights, feature_names=None):
        """
        Initialize the EntropyScore Class.

        Parameters:
        - data: np.ndarray of shape (n_samples, n_features), with categorical values
        - weights: np.ndarray of shape (n_samples,), sample weights
        - feature_names: list of str of length n_features (optional)
        """
        self.data = data
        self.weights = weights
        self.n_features = data.shape[1]
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(self.n_features)]
        self.entropy_scores = self._compute_entropy_scores()
        self.cluster_labels = self._make_clustering(self.entropy_scores)

    def _compute_entropy_scores(self):
        """
        Compute normalized weighted entropy for each categorical feature.

        Returns:
        - np.ndarray of shape (n_features,), entropy scores
        """
        n_samples, n_features = self.data.shape
        entropies = np.zeros(n_features)

        for j in range(n_features):
            freq = defaultdict(float)
            for i in range(n_samples):
                value = self.data[i, j]
                freq[value] += self.weights[i]

            total_weight = sum(freq.values())
            probs = np.array([w / total_weight for w in freq.values() if w > 0])
            N = len(probs)

            if N > 1:
                log_base_N = np.log(N)
                entropies[j] = 1+np.sum(probs * np.log(probs)) / log_base_N
            else:
                entropies[j] = 0.0

        return entropies

    def _make_clustering(self, entropy_scores):
        """
        Cluster features using KMeans on entropy scores.

        Parameters:
        - entropy_scores: np.ndarray of shape (n_features,), entropy scores

        Returns:
        - labels: np.ndarray of shape (n_features,), cluster labels
        """
        kmeans = KMeans(n_clusters=2, n_init=23, random_state=42)
        kmeans.fit(entropy_scores.reshape(-1, 1))

        clue_centers_ = np.array([[kmeans.cluster_centers_.min()],
                                  [kmeans.cluster_centers_.max()]])
        clue_dist = euclidean_distances(entropy_scores.reshape(-1, 1), clue_centers_)
        labels = np.argmin(clue_dist, axis=-1)
        return labels

    def stability_validation(self):
        """
        Validate the clustering using Leave-One-Out cross-validation.

        Returns:
        - float: validation score (1 - mean MCC)
        """
        val_scores = []
        baseline_labels = self.cluster_labels
        loo = LeaveOneOut()

        for train_idx, _ in loo.split(self.entropy_scores):
            train_scores = self.entropy_scores[train_idx]
            loo_labels = self._make_clustering(train_scores)
            val = accuracy_score(baseline_labels[train_idx], loo_labels)
            val_scores.append(val)

        #validation_score
        validation_score = lambda values: np.mean(values)
        
        # jackknife statistic
        jack_stat = jackknife_stats(np.array(val_scores), validation_score)
        
        return {"validation_score": jack_stat[0], 
                "confidence_interval": jack_stat[-1]}

    
    def internal_validation(self, nfolds= 10):
        """
        Internal validation using the K-fold cross-validation

        Returns:
        - float: validation score (1 - mean MCC)
        """
        val_scores = []
        baseline_labels = self.cluster_labels
        validator = KFold(n_splits=nfolds)

        for train_idx, _ in validator.split(self.entropy_scores):
            train_scores = self.entropy_scores[train_idx]
            loo_labels = self._make_clustering(train_scores)
            val = accuracy_score(baseline_labels[train_idx], loo_labels)
            val_scores.append(val)

        #validation_score
        validation_score = lambda values: np.mean(values)
        
        # jackknife statistic
        jack_stat = jackknife_stats(np.array(val_scores), validation_score)
        
        return {"validation_score": jack_stat[0], 
                "confidence_interval": jack_stat[-1]}

    def get_entropy_dataframe(self):
        """
        Return a DataFrame with feature names, entropy scores, and cluster labels.

        Returns:
        - pd.DataFrame with columns: 'feature', 'entropy_score', 'cluster'
        """
        df = pd.DataFrame({
            "Variable_Name": self.feature_names,
            "entropy_score": self.entropy_scores,
            "cluster": ["typical" if lbl == 0 else "atypical" for lbl in self.cluster_labels]
        })
        return df.sort_values(by="entropy_score", ascending=False).reset_index(drop=True)


# ==========================================
# CLASS: OutlierDetection
# ------------------------------------------
# Purpose:
# A unified class to detect and analyze outliers in tabular datasets
# using three alternative feature-extraction strategies:
#   - PCA  (linear)
#   - KPCA (non-linear via Hamming kernel)
#   - Autoencoder (non-linear neural representation)
#
# The class supports:
#   - Weighted data (e.g., survey weights)
#   - Clustering and medoid identification
#   - Leave-One-Out and k-fold stability validation
#   - Variable importance estimation through permutation
# ==========================================

class OutlierDetection:
    def __init__(self, data, weights, method, feature_names=None, variance_explained=95e-2):
        """
        Initialize the OutlierDetection instance.

        Args:
            data (ndarray): 2D array of shape (n_samples, n_features)
            weights (ndarray): array of sample weights, shape (n_samples,)
            method (str): chosen method for feature representation
                          options: 'PCA', 'KPCA', 'AE'
            feature_names (list[str], optional): names of variables
            variance_explained (float): threshold for cumulative explained variance
                                        (used to determine number of retained components)
        """

        # Store input data and associated parameters
        self.data = data
        self.weights = weights
        self.n_features = data.shape[1]
        self.n_samples = data.shape[0]
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(self.n_features)]
        self.method = method
        self.var_explained = variance_explained

        # Step 1: Compute anomaly scores based on the selected method
        self.outlier_scores = self._compute_anomaly_scores()

        # Step 2: Apply clustering on the outlier scores (KMeans with 2 clusters)
        self.cluster_labels = self._make_clustering(self.outlier_scores)

        # Step 3: Identify the medoid (most representative outlier)
        self.medoid = self._find_medoid()

        self.utils = Utils() 
        
    # ---------------------------------------------------------------
    # AUTOENCODER SUBNETWORK
    # ---------------------------------------------------------------
    def _build_autoencoder(self, input_dim):
        """
        Build and compile a simple feedforward autoencoder.

        Architecture:
            Input -> Dense(tanh) -> Dense(tanh) -> Dense(tanh) -> Output(softplus)
        """
        input_layer = tf.keras.layers.Input(shape=(input_dim,))

        # Encoder: compress representation
        encoded = tf.keras.layers.Dense(input_dim // 2, activation="tanh")(input_layer)
        encoded = tf.keras.layers.Dense(input_dim // 4, activation="tanh")(encoded)

        # Decoder: reconstruct input
        decoded = tf.keras.layers.Dense(input_dim // 2, activation="tanh")(encoded)
        output_layer = tf.keras.layers.Dense(input_dim, activation="softplus")(decoded)

        # Define and compile autoencoder
        autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer="adam", loss="mse")

        return autoencoder


    # ---------------------------------------------------------------
    # KERNEL CENTERING (Weighted)
    # ---------------------------------------------------------------
    def weighted_center_kernel(self, K, weights):
        """
        Center a kernel (Gram) matrix using weighted mean correction.

        This ensures that the kernel matrix has zero mean in feature space,
        accounting for the presence of sampling weights.
        """
        weights = weights.reshape(-1, 1)
        one = np.ones((K.shape[0], 1))

        # Weighted centering
        K_weights_row = K @ weights
        K_weights_col = weights.T @ K
        K_weighted_mean = weights.T @ K @ weights

        K_centered = K - K_weights_row - K_weights_col + K_weighted_mean
        return K_centered


    # ---------------------------------------------------------------
    # WEIGHTED HAMMING KERNEL MATRIX
    # ---------------------------------------------------------------
    def _get_correct_Gram_matrix(self, data, weights, gamma=1):
        """
        Compute a weighted Hamming kernel and center it.

        Args:
            data (ndarray): binary or categorical data
            weights (ndarray): sample weights
            gamma (float): kernel bandwidth parameter
        """
        Ham_K = cdist(data, data, metric="hamming")   # Pairwise Hamming distances
        Ham_K = np.exp(-gamma * Ham_K)                # Convert distances to similarities
        Ham_K = self.weighted_center_kernel(Ham_K, weights)
        return Ham_K


    # ---------------------------------------------------------------
    # COMPUTE ANOMALY SCORES
    # ---------------------------------------------------------------
    def _compute_anomaly_scores(self):
        """
        Compute an outlier (anomaly) score for each sample depending on the chosen method:
            - PCA: based on reconstruction residuals from discarded components
            - KPCA: non-linear analogue using a kernel matrix
            - AE: reconstruction error from an autoencoder
        """

        if self.method == "PCA":
            # Fit PCA model
            self.pca_fr = PCA(n_components=self.n_features)
            self.pca_fr.fit(self.data)

            # Determine number of components explaining desired variance
            cum_var = np.cumsum(self.pca_fr.explained_variance_ratio_)
            self.n_leftover_components = np.argmax(cum_var > self.var_explained) + 1

            # Use the residual components as anomaly indicators
            scores = self.pca_fr.transform(self.data)[:, self.n_leftover_components:]

        elif self.method == "KPCA":
            # Compute weighted kernel PCA
            Ham_K = self._get_correct_Gram_matrix(self.data, self.weights)
            self.kpca_fr = KernelPCA(n_components=self.n_samples,
                                     kernel="precomputed",
                                     fit_inverse_transform=False)
            self.kpca_fr.fit(Ham_K)

            # Determine variance explained by eigenvalues
            eigvals = self.kpca_fr.eigenvalues_
            cum_var = np.cumsum(eigvals / eigvals.sum())
            self.n_leftover_components = np.argmax(cum_var > self.var_explained) + 1

            scores = self.kpca_fr.transform(Ham_K)[:, self.n_leftover_components:]

        elif self.method == "AE":
            # Build and train autoencoder
            self.autoencoder = self._build_autoencoder(self.n_features)
            self.autoencoder.fit(self.data, self.data,
                                 sample_weight=self.weights,
                                 epochs=500, batch_size=16, verbose=0)

            # Reconstruction error as anomaly measure
            reconstructed = self.autoencoder.predict(self.data, verbose=0)
            scores = np.square(self.data - reconstructed)

        else:
            raise ValueError("Invalid method. Choose 'PCA', 'KPCA', or 'AE'.")

        # Compute sample-level anomaly score (sum of squared residuals)
        return np.einsum("ij, ij -> i", scores, scores)


    # ---------------------------------------------------------------
    # CLUSTERING OF OUTLIER SCORES
    # ---------------------------------------------------------------
    def _make_clustering(self, outlier_scores):
        """
        Cluster anomaly scores into 2 groups using KMeans.
        The group with higher mean score is interpreted as the 'outlier' cluster.
        """
        self.kmeans = KMeans(n_clusters=2, n_init=23, random_state=42)
        self.kmeans.fit(outlier_scores.reshape(-1, 1))

        # Identify which cluster corresponds to 'normal' vs 'outlier'
        clue_centers_ = np.array([[self.kmeans.cluster_centers_.min()],
                                  [self.kmeans.cluster_centers_.max()]])
        clue_dist = euclidean_distances(outlier_scores.reshape(-1, 1), clue_centers_)

        return np.argmin(clue_dist, axis=-1)


    # ---------------------------------------------------------------
    # SUBCLUSTERING OF OUTLIERS
    # ---------------------------------------------------------------
    def _make_subclustering(self, max_n_clusters):
        """
        Perform spectral subclustering on detected outliers
        to identify distinct anomaly categories.
        """
        outliers = self.data[self.cluster_labels == 1]

        # Compute pairwise similarity
        distance = cdist(outliers, outliers, "hamming")
        affinity = np.exp(-distance)

        n_clue = np.arange(4, max_n_clusters, 1)
        silh_score = []

        # Optimize cluster count by silhouette score
        for k in n_clue:
            sclue = SpectralClustering(n_clusters=k, affinity="precomputed")
            sclue.fit(affinity)
            silh_score.append(silhouette_score(distance, sclue.labels_, metric="precomputed"))

        optimal_nclue = n_clue[np.argmax(silh_score)]
        SPCLUE = SpectralClustering(n_clusters=optimal_nclue, affinity="precomputed")
        labels = SPCLUE.fit_predict(affinity)

        print("Silhouette_Score:", silhouette_score(distance, SPCLUE.labels_, metric="precomputed"))

        # Identify medoid for each subcluster
        categories = []
        for label in np.unique(labels):
            cluster_data = outliers[labels == label]
            intra_dist = cdist(cluster_data, cluster_data, "hamming")
            idx_medoid = np.argmin(np.sum(intra_dist, axis=-1))
            categories.append(cluster_data[idx_medoid])

        # Construct output DataFrame summarizing cluster medoids
        dict_ = {"Variable_Name": self.feature_names}
        for i, cat in enumerate(categories):
            dict_[f"Catg_{i+1}"] = cat

        return pd.DataFrame(dict_)


    # ---------------------------------------------------------------
    # FIND MEDOID (Most Central Outlier)
    # ---------------------------------------------------------------
    def _find_medoid(self):
        """
        Identify the most representative outlier (medoid)
        based on minimal average Hamming distance within the outlier cluster.
        """
        outliers = self.data[self.cluster_labels == 1]
        intra_dist = cdist(outliers, outliers, "hamming")
        idx_medoid = np.argmin(np.sum(intra_dist, axis=-1))
        medoid_ = outliers[idx_medoid]
        return pd.DataFrame({"Variable_Name": self.feature_names, "Code": medoid_})


    # ---------------------------------------------------------------
    # LEAVE-ONE-OUT STABILITY VALIDATION
    # ---------------------------------------------------------------
    def validate_stability(self):
        """
        Evaluate clustering stability using Leave-One-Out cross-validation
        and the Matthews correlation coefficient (MCC) as the metric.
        Returns 1 - mean(MCC), representing clustering instability.
        """
        val_scores = []
        baseline_labels = self.cluster_labels
        loo = LeaveOneOut()

        for train_idx, _ in loo.split(self.outlier_scores):
            train_scores = self.outlier_scores[train_idx]
            loo_labels = self._make_clustering(train_scores)
            val_scores.append(matthews_corrcoef(baseline_labels[train_idx], loo_labels))

        #validation_score
        validation_score = lambda values: np.mean(values)
        
        # jackknife statistic
        jack_stat = jackknife_stats(np.array(val_scores), validation_score)
        
        return {"validation_score": jack_stat[0], 
                "confidence_interval": jack_stat[-1]}


    # ---------------------------------------------------------------
    # K-FOLD INTERNAL VALIDATION
    # ---------------------------------------------------------------
    def internal_validation(self, n_repeat=10):
        """
        Assess clustering consistency using k-fold cross-validation
        with jackknife bias correction applied to MCC estimates.
        """
        baseline_labels = self.cluster_labels
        mcc = []
        n_features = self.data.shape[1]

        kfold = KFold(10, shuffle=True, random_state=6)
        kfold_labels = np.zeros(baseline_labels.size)

        idx = 0
        for train_idx, val_idx in kfold.split(self.outlier_scores):
            idx += 1
            train_scores = self.outlier_scores[train_idx]
            val_scores = self.outlier_scores[val_idx]

            # Train clustering on training fold
            kmeans = KMeans(n_clusters=2, n_init=23, random_state=42)
            kmeans.fit(train_scores.reshape(-1, 1))

            # Assign labels to validation fold using cluster centroids
            clue_centers_ = np.array([[kmeans.cluster_centers_.min()],
                                      [kmeans.cluster_centers_.max()]])
            val_labeling = np.argmin(
                euclidean_distances(val_scores.reshape(-1, 1), clue_centers_),
                axis=-1
            )

            kfold_labels[val_idx] = val_labeling
            print("Fold:", idx)

        # Apply jackknife bias correction to MCC estimates
        mcc_biascorr, mcc_mean, mcc_se = self.utils.jackknife_bias_corrected_xy(
            kfold_labels, baseline_labels, matthews_corrcoef)

        return {"mcc_mean": mcc_biascorr, "mcc_sem": mcc_se}


    # ---------------------------------------------------------------
    # VARIABLE IMPORTANCE VIA PERMUTATION
    # ---------------------------------------------------------------
    def varibale_importance(self, n_repeat=30):
        """
        Estimate variable importance by measuring the decrease in
        clustering accuracy after permuting each feature multiple times.
        """
        val, val_err = [], []

        for nfeat in range(self.data.shape[1]):
            Xcopy = self.data.copy()
            case_specific_val = []

            for _ in range(n_repeat):
                np.random.shuffle(Xcopy[:, nfeat])

                # Recompute anomaly scores for permuted data
                if self.method == "PCA":
                    score = self.pca_fr.transform(Xcopy)[:, self.n_leftover_components:]
                elif self.method == "KPCA":
                    Kcopy = self._get_correct_Gram_matrix(Xcopy, self.weights)
                    score = self.kpca_fr.transform(Kcopy)[:, self.n_leftover_components:]
                elif self.method == "AE":
                    reconstructed = self.autoencoder.predict(Xcopy, verbose=0)
                    score = np.square(Xcopy - reconstructed)
                else:
                    raise ValueError("Invalid method.")

                outlier_score = np.einsum("ij, ij -> i", score, score)
                new_labels = self.kmeans.predict(outlier_score.reshape(-1, 1))

                # Measure how similar new labels are to baseline clustering
                acc = accuracy_score(new_labels, self.cluster_labels)
                case_specific_val.append(acc)

            val.append(np.mean(case_specific_val))
            val_err.append(sem(case_specific_val))

        return {
            "importance_average": 1 - np.array(val),
            "importance_sem": val_err
        }


    # ---------------------------------------------------------------
    # BUILD A RANKED DATAFRAME OF FEATURE IMPORTANCE
    # ---------------------------------------------------------------
    def get_importance_dataframe(self):
        """
        Create a sorted DataFrame summarizing average feature importance.
        """
        importance_score = self.varibale_importance()
        score_col_name = f"Average Importance ({self.method})"

        df = pd.DataFrame({
            "Variable_Name": self.feature_names,
            score_col_name: list(importance_score["importance_average"])
        })

        return df.sort_values(by=score_col_name, ascending=False).reset_index(drop=True)