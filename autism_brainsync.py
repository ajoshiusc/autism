
# We start by downloading the data from Internet

from problem import get_train_data
data_train, labels_train = get_train_data()

data_train.head()
print(labels_train)
print('Number of subjects in the training tests: {}'.format(labels_train.size))
data_train_participants = data_train[[col for col in data_train.columns if col.startswith('participants')]]
data_train_participants.head()



data_train_functional = data_train[[col for col in data_train.columns if col.startswith('fmri')]]
data_train_functional.head()


# Unlike the anatomical and participants data, the available data are filename to CSV files in which the time-series information are stored. We show in the next section how to read and extract meaningful information from those data.

# Similarly to the anatomical data, the column `fmri_select` gives information about the manual quality check.
data_train_functional['fmri_select'].head()


import scipy as sp
import numpy as np
"""
Created on Tue Jul 11 22:42:56 2017

 """


def normalizeData(pre_signal):
    """
     normed_signal, mean_vector, std_vector = normalizeData(pre_signal)
     This function normalizes the input signal to have 0 mean and unit
     variance in time.
     pre_signal: Time x Original Vertices data
     normed_signal: Normalized (Time x Vertices) signal
     mean_vector: 1 x Vertices mean for each time series
     norm_vector : 1 x Vertices norm for each time series
    """

    if sp.any(sp.isnan(pre_signal)):
        print('there are NaNs in the data matrix, making them zero')

    pre_signal[sp.isnan(pre_signal)] = 0
    mean_vector = sp.mean(pre_signal, axis=0, keepdims=True)
    normed_signal = pre_signal - mean_vector
    norm_vector = sp.linalg.norm(normed_signal, axis=0, keepdims=True)
    norm_vector[norm_vector == 0] = 1e-116
    normed_signal = normed_signal / norm_vector

    return normed_signal, mean_vector, norm_vector


def brainSync(X, Y):
    """
   Input:
       X - Time series of the reference data (Time x Vertex) \n
       Y - Time series of the subject data (Time x Vertex)

   Output:
       Y2 - Synced subject data (Time x Vertex)\n
       R - The orthogonal rotation matrix (Time x Time)

   Please cite the following publication:
       AA Joshi, M Chong, RM Leahy, BrainSync: An Orthogonal Transformation
       for Synchronization of fMRI Data Across Subjects, Proc. MICCAI 2017,
       in press.
       """
    if X.shape[0] > X.shape[1]:
        print('The input is possibly transposed. Please check to make sure \
that the input is time x vertices!')

    C = np.dot(X, Y.T)
    C[~np.isfinite(C)] = 0.0000001
    U, _, V = sp.linalg.svd(C)
    R = np.dot(U, V)
    Y2 = np.dot(R, Y)
    return Y2, R

# The testing data can be loaded similarly as follows:

from problem import get_test_data

data_test, labels_test = get_test_data()
data_test.head()
a=data_test['fmri_basc197']

# ### Evaluation
# The framework is evaluated with a cross-validation approach. The metrics used are the AUC under the ROC and the accuracy.
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from problem import get_cv

def evaluation(X, y):
    pipe = make_pipeline(FeatureExtractor(), Classifier())
    cv = get_cv(X, y)
    results = cross_validate(pipe, X, y, scoring=['roc_auc', 'accuracy'], cv=cv,
                             verbose=1, return_train_score=True,
                             n_jobs=10)
    return results

# ### Going further: using fMRI-derived features
# From the framework illustrated in the figure above, steps 1 to 2 already have been computed during some preprocessing and are the data given during this challenge. Therefore, our feature extractor will implement the step #3 which correspond to the extraction of functional connectivity features. Step 4 is identical to the pipeline presented for the anatomy with a standard scaler followed by a logistic regression classifier.
# We pointed out that the available feature for fMRI are filename to the time-series. In order to limit the amount of data to be downloaded, we provide a fetcher `fetch_fmri_time_series()` to download only the time-series linked to a specific atlases.

from download_data import fetch_fmri_time_series
fetch_fmri_time_series(atlas='basc197')

# You can download all atlases at once by passing `atlas='all'`. It is also possible to execute the file as a script `python download_data.py all`.

# In the `FeatureExtractor` below, we first only select the filename related to the MSDL time-series data. We create a `FunctionTransformer` which will read on-the-fly the time-series from the CSV file and store them into a numpy array.
# Those series will be used to extract the functional connectivity matrices which will be used later in the classifier.


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from nilearn.connectome import ConnectivityMeasure
TRUNC = 200


def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    a = np.array([pd.read_csv(subject_filename,
                              header=None).values
                  for subject_filename in fmri_filenames])
    Z = np.zeros((500, a[0].shape[1]))

    for i in range(len(a)):
        Z[:a[i].shape[0], :] = a[i]
        a[i] = Z[:TRUNC, ]
        Z = 0*Z

    return a


class BrainSyncTransform(BaseEstimator, TransformerMixin):
        def __init__(self, refdata=np.array([0])):
            self.refdata = refdata

        def fit(self, X, y=None):
            self.refdata = y

            return self

        def transform(self, X, y=None):
            for ind in range(len(X)):
                X[ind], _ = brainSync(X=self.refdata, Y=X[ind])
                X[ind] = X[ind].flatten()

            X = np.vstack(X)

            return X


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.ref = 0
        self.refdata = np.array([0])
        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            BrainSyncTransform())

    def fit(self, X_df, y):
        # get only the time series for the MSDL atlas

        fmri_filenames = X_df['fmri_basc197']
        if self.ref == 0:
            self.ref = fmri_filenames[fmri_filenames.index[0]]

        aa = pd.read_csv(self.ref, header=None).values
        Z = np.zeros((500, aa.shape[1]))
        Z[:aa.shape[0], :] = aa
        self.refdata = Z[:TRUNC, ]
        self.refdata, _, _ = normalizeData(self.refdata)
        self.transformer_fmri.fit(fmri_filenames, self.refdata)
        return self

    def transform(self, X_df):
        fmri_filenames = X_df['fmri_basc197']
        return self.transformer_fmri.transform(fmri_filenames)



from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression())

    refdata = 0

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


results = evaluation(data_train, labels_train)

print("Training score ROC-AUC: {:.3f} +- {:.3f}".format(np.mean(results['train_roc_auc']),
                                                        np.std(results['train_roc_auc'])))
print("Validation score ROC-AUC: {:.3f} +- {:.3f} \n".format(np.mean(results['test_roc_auc']),
                                                          np.std(results['test_roc_auc'])))

print("Training score accuracy: {:.3f} +- {:.3f}".format(np.mean(results['train_accuracy']),
                                                         np.std(results['train_accuracy'])))
print("Validation score accuracy: {:.3f} +- {:.3f}".format(np.mean(results['test_accuracy']),
                                                           np.std(results['test_accuracy'])))
