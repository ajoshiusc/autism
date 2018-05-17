
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

# The testing data can be loaded similarly as follows:

from problem import get_test_data

data_test, labels_test = get_test_data()
data_test.head()
print(labels_test)

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
fetch_fmri_time_series(atlas='msdl')

# You can download all atlases at once by passing `atlas='all'`. It is also possible to execute the file as a script `python download_data.py all`.

# In the `FeatureExtractor` below, we first only select the filename related to the MSDL time-series data. We create a `FunctionTransformer` which will read on-the-fly the time-series from the CSV file and store them into a numpy array.
# Those series will be used to extract the functional connectivity matrices which will be used later in the classifier.


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from nilearn.connectome import ConnectivityMeasure


def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return np.array([pd.read_csv(subject_filename,
                                 header=None).values
                     for subject_filename in fmri_filenames])


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # make a transformer which will load the time series and compute the
        # connectome matrix
        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind='tangent', vectorize=True))
        
    def fit(self, X_df, y):
        # get only the time series for the MSDL atlas
        fmri_filenames = X_df['fmri_msdl']
        self.transformer_fmri.fit(fmri_filenames, y)
        return self

    def transform(self, X_df):
        fmri_filenames = X_df['fmri_msdl']
        return self.transformer_fmri.transform(fmri_filenames)


# In[23]:


from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.))

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
