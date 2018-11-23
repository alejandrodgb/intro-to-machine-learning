#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

#from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

print('Libraries imported')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### This list of features is the final list of features after cleaning.
features_list = ['poi', 'salary', 'total_payments', 'bonus', 
                'total_stock_value', 'expenses',
                'exercised_stock_options','other','restricted_stock',
                'from_poi_to_this_person','shared_receipt_with_poi',
                'deferred_income','long_term_incentive','to_poi_proportion',
                'bonus_to_salary']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# load data to dataframe for ease of modification
df = pd.DataFrame.from_dict(data_dict, orient='index')
df.reset_index(inplace=True)
df.rename(columns={'index':'full_name'}, inplace=True)

df = pd.DataFrame.from_dict(data_dict, orient='index')
df.reset_index(inplace=True)
df.rename(columns={'index':'full_name'}, inplace=True)

# classifying the columns and modifying dataframe
continuous_features = ['salary', 'deferral_payments', 'total_payments', 
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees', 'to_messages', 
                       'from_poi_to_this_person', 'from_messages', 
                       'from_this_person_to_poi', 'shared_receipt_with_poi']

email_features= ['to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

for feat in continuous_features:
    df[feat]=df[feat].astype(float)

# invalid rows. See poi_working_notebook.ipynb notebook for cleaning analysis
rows_to_remove = [84,130,127]
df.drop(rows_to_remove, axis=0, inplace=True)

# correct incorrect entry
nan_cols = [1,6,18,9,3,5,15,21]
valid_cols=[11,4,13,20,8,10]
cols_values=[137864, 137864, 15456290, 2604490, -2604490, 15456290]
df.iloc[11,valid_cols]=cols_values
df.iloc[11,nan_cols]=np.nan

print('Dataframe loaded and corrected')
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# proportion of received messages from poi out of total received messages
# and sent messages to poi out of total sent messages
df['from_poi_proportion']=df.from_poi_to_this_person/df.to_messages
df['to_poi_proportion']=df.from_this_person_to_poi/df.from_messages

# bonus to salary ratio
df['bonus_to_salary']=df.bonus/df.salary

# restricted stock as a proportion of total stock value
df['restricted_to_total_stock']=df.restricted_stock/df.total_stock_value

df.loc[df[df.email_address=='NaN'].index,'email_address']=np.nan
df['poi']=df.poi.apply(lambda x: 1 if x else 0)
data_dict = df.set_index('full_name').to_dict(orient='index')

print('New features created')
print('data_dict created')

# convert np.nan to 'NaN' for compatibility with tester
for k,v in data_dict.items():
    for sk,sv in v.items():
        if sv!=sv:
            data_dict[k][sk]='NaN'

my_dataset = data_dict

print('data_dict to my_dataset')
### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

features = np.array(df[features_list].fillna(0).iloc[:,1:])
labels = np.array(df[features_list].fillna(0).iloc[:,0])

print('Features and labels created')

### Task 4 & 5: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
### Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print('Start training')

# Example starting point. Try investigating other evaluation techniques!
from sklearn import model_selection
from sklearn import ensemble
features_train, features_test, labels_train, labels_test = \
model_selection.train_test_split(features, labels, test_size=0.3, 
                                 random_state=42)


params_grid={'n_estimators':[25,50,100,150,200,300]}
clf = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), 
                                   params_grid, return_train_score=True)
clf.fit(features_train, labels_train)
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
print('Process finalised')