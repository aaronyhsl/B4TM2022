# import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# load data
data = pd.read_csv("../data/Train_call.txt", delimiter="\t")
labels = pd.read_csv("../data/Train_clinical.txt", delimiter="\t")

# transpose the dataframe
data = data.transpose()
data = data[4:]
data = data.reset_index()
data.rename(columns={"index": "Sample"}, inplace=True)
data.head()

labels.head()

# Combine labels and instances
combined = pd.merge(data, labels, on="Sample",)
combined

X = combined.iloc[:, 1:2835]
y = combined["Subgroup"]

combined.head()

# check if class lables are balanced or not
y.value_counts()

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

###===================###
### Feature selection ###
###===================###

### Mutual information ###

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib.pyplot import figure
from matplotlib import pyplot

mi_score = mutual_info_classif(X_train, y_train, random_state=100)
pd.DataFrame(mi_score)

# what are the MI scores for each feature
for i in range(len(mi_score)):
    print("Feature %d: %f" % (i, mi_score[i]))

# plot the scores
figure(figsize=(10, 8))
pyplot.bar([i for i in range(len(mi_score))], mi_score)
plt.title("Mutual information scores among all features")
plt.xlabel("All features")
plt.ylabel("Mutual information scores")
pyplot.show()


# calculate the mean score
fs_mutinfo_all_df = pd.DataFrame(mi_score)
fs_mutinfo_all_df.rename(columns={0: "Scores"}, inplace=True)
mean_mutinfo = fs_mutinfo_all_df["Scores"].mean()
print("Mean score of mutual information:", mean_mutinfo)

# select features higher or equal to the threshold
fs_mutinfo = fs_mutinfo_all_df.loc[fs_mutinfo_all_df["Scores"] >= 0.18]

fs_mutinfo.sort_values(by="Scores", ascending=False).to_csv("MI_scores.csv")

# get index of selected features
selected_features_list = list(fs_mutinfo.T.columns)

# fit X_train and X_test with selected features
X_train_fs_mutinfo = X_train.filter(items=selected_features_list, axis=1)
X_test_fs_mutinfo = X_test.filter(items=selected_features_list, axis=1)
X_train_fs_mutinfo


###=========================###
### Nested Cross-validation ###
###=========================###

# - perform inner loop again on the whole training data to find the optimal hyperparameters
# - perform nested cross-validation to validate the model building strategy

### XGBoost ###

from sklearn.model_selection import cross_val_score, KFold

# parameters for xgb
param_grid_xgb = {
    "max_depth": range(3, 10),
    "min_child_weight": range(1, 6),
    "gamma": [i / 10.0 for i in range(0, 5)],
    # 'subsample': [i/10.0 for i in range(6,10)],
    # 'colsample_bytree': [i/10.0 for i in range(6,10)],
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 100],
}

# define xgb model
model_to_tune_xgb = XGBClassifier(
    random_state=42, num_class=3, objective="multi:softmax"
)

# Declare the inner and outer cross-validation strategies
inner_cv_xgb = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv_xgb = KFold(n_splits=3, shuffle=True, random_state=42)

# Inner cross-validation for parameter search
model_xgb = GridSearchCV(
    estimator=model_to_tune_xgb, param_grid=param_grid_xgb, cv=inner_cv_xgb, n_jobs=2
)

model_xgb.fit(X_train_fs_mutinfo, lc_y_train)

# Outer cross-validation to compute the testing score
test_score_xgb = cross_val_score(
    model_xgb, X_train_fs_mutinfo, lc_y_train, cv=outer_cv_xgb, n_jobs=2
)

print(
    f"The mean score using nested cross-validation is: "
    f"{test_score_xgb.mean():.3f} +/- {test_score_xgb.std():.3f}"
)
print(f"The best parameters found are: {model_xgb.best_params_}")


### Random Forest ###

# parameters for random forest
param_grid_rf = {
    "max_depth": range(3, 10),
    "n_estimators": [1, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "bootstrap": [True, False],
}

# define random forest model
model_to_tune_rf = RandomForestClassifier(random_state=42)

# Declare the inner and outer cross-validation strategies
inner_cv_rf = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv_rf = KFold(n_splits=3, shuffle=True, random_state=42)

# Inner cross-validation for parameter search
model_rf = GridSearchCV(
    estimator=model_to_tune_rf, param_grid=param_grid_rf, cv=inner_cv_rf, n_jobs=2
)

model_rf.fit(X_train_fs_mutinfo, y_train)

# Outer cross-validation to compute the testing score
test_score_rf = cross_val_score(
    model_rf, X_train_fs_mutinfo, y_train, cv=outer_cv_rf, n_jobs=2
)

print(
    f"The mean score using nested cross-validation is: "
    f"{test_score_rf.mean():.3f} +/- {test_score_rf.std():.3f}"
)
print(f"The best parameters found are: {model_rf.best_params_}")

### Logistic Regression ###

# parameters for logistic regression

param_grid_lr = {
    "penalty": ["l1", "l2"],
    "C": np.logspace(-4, 4, 20),
    "solver": ["lbfgs"],
}

# define logistic regression model
model_to_tune_lr = LogisticRegression(multi_class="multinomial")

# Declare the inner and outer cross-validation strategies
inner_cv_lr = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv_lr = KFold(n_splits=3, shuffle=True, random_state=42)

# Inner cross-validation for parameter search
model_lr = GridSearchCV(
    estimator=model_to_tune_lr, param_grid=param_grid_lr, cv=inner_cv_lr, n_jobs=2
)

model_lr.fit(X_train_fs_mutinfo, y_train)

# Outer cross-validation to compute the testing score
test_score_lr = cross_val_score(
    model_lr, X_train_fs_mutinfo, y_train, cv=outer_cv_lr, n_jobs=2
)

print(
    f"The mean score using nested cross-validation is: "
    f"{test_score_lr.mean():.3f} +/- {test_score_lr.std():.3f}"
)
print(f"The best parameters found are: {model_lr.best_params_}")


# make a dataframe containing all scores from nested cv
all_scores = {
    "XGBoost": test_score_xgb,
    "Random Forest": test_score_rf,
    "Logistic Regression": test_score_lr,
}
all_scores = pd.DataFrame(all_scores)

import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
all_scores.plot.box(color=color, vert=False, showmeans=True)
plt.xlabel("Accuracy")
plt.title("Comparison of mean accuracy on the nested cross-validation")

###=======================###
###     Gridsearch CV     ###
###=======================###

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import loguniform
from sklearn import metrics

### XGBoost ###

# define model
model_xgb = XGBClassifier(random_state=42, num_class=3, objective="multi:softmax")

# define evaluation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space_xgb = dict()
space_xgb["max_depth"] = range(3, 10)
space_xgb["min_child_weight"] = range(1, 6)
space_xgb["gamma"] = [i / 10.0 for i in range(0, 5)]
# space_xgb['subsample'] = [i/10.0 for i in range(6,10)]
# space_xgb['colsample_bytree'] = [i/10.0 for i in range(6,10)]
space_xgb["reg_alpha"] = [1e-5, 1e-2, 0.1, 1, 100]

# define search_xgb
search_xgb = GridSearchCV(model_xgb, space_xgb, scoring="accuracy", n_jobs=2, cv=5)

# execute search_xgb
result_xgb = search_xgb.fit(X_train_fs_mutinfo, lc_y_train)

# summarize result_xgb
# best_score_ is the mean cross validated scores of the best estimator
print("Best Score: %s" % result_xgb.best_score_)
print("Best Hyperparameters: %s" % result_xgb.best_params_)

# print winning set of hyperparameters
from pprint import pprint

pprint(result_xgb.best_estimator_.get_params())
result_xgb.best_estimator_.feature_importances_

### Random Forest ###

from sklearn.ensemble import RandomForestClassifier

# define model
model_rf = RandomForestClassifier(random_state=42)

# define evaluation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space_rf = dict()
space_rf["max_depth"] = range(3, 10)
space_rf["n_estimators"] = [1, 5, 10, 15, 20]
space_rf["min_samples_leaf"] = [1, 2, 3, 4, 5]
space_rf["bootstrap"] = [True, False]

# define search_rfspace_rf
search_space_rf = GridSearchCV(
    model_rf, space_rf, scoring="accuracy", n_jobs=-1, cv=5, verbose=True
)

# execute search_rfspace_rf
result_rf = search_space_rf.fit(X_train_fs_mutinfo, y_train)

# summarize result_rfspace_rf
print("Best Score: %s" % result_rf.best_score_)
print("Best Hyperparameters: %s" % result_rf.best_params_)

# print winning set of hyperparameters
from pprint import pprint

pprint(result_rf.best_estimator_.get_params())

### Logistic Regression ###
from sklearn.linear_model import LogisticRegression

# define model
model_lr = LogisticRegression(multi_class="multinomial")

# define evaluation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space_lr = dict()
space_lr["penalty"] = ["l1", "l2"]
space_lr["C"] = np.logspace(-4, 4, 20)
space_lr["solver"] = ["lbfgs"]


# define search_lrspace_lrspace_lr
search_space_lr = GridSearchCV(
    model_lr, space_lr, scoring="accuracy", n_jobs=-1, cv=5, verbose=True
)

# execute search_lrspace_lrspace_lr
result_lr = search_space_lr.fit(X_train_fs_mutinfo, y_train)

# summarize result_lrspace_lrspace_lr
print("Best Score: %s" % result_lr.best_score_)
print("Best Hyperparameters: %s" % result_lr.best_params_)

# print winning set of hyperparameters
from pprint import pprint

pprint(result_lr.best_estimator_.get_params())

### Get final optimal models ###

### final optimal models train on all train set

# XGBoost
opt_xgb = result_xgb.best_estimator_
final_opt_xgb = opt_xgb.fit(X_train_fs_mutinfo, lc_y_train)

# Random Forest
opt_rf = result_rf.best_estimator_
final_opt_rf = opt_rf.fit(X_train_fs_mutinfo, y_train)

# Logistic Regression
opt_lr = result_lr.best_estimator_
final_opt_lr = opt_lr.fit(X_train_fs_mutinfo, y_train)


###====================================###
### Model evaluation (on training set) ###
###====================================###

from xgboost import plot_importance

X_train_fs_mutinfo.columns

## Aligned feature names

X_train_fs_mutinfo_list = list(X_train_fs_mutinfo.columns)

# Random Forest
feature_importances_rf = pd.Series(opt_rf.feature_importances_)
feature_importances_rf.index = X_train_fs_mutinfo_list

# Logist Regression
feature_importances_lr = pd.Series(opt_lr.coef_[0])
feature_importances_lr.index = X_train_fs_mutinfo_list

# Feature importances

# XGBoost
feature_importances_xgb = pd.Series(
    opt_xgb.get_booster().get_score(importance_type="gain")
)
feature_importances_xgb.sort_values(ascending=False).to_csv(
    "XGB_feature_importance.csv"
)

# Random Forest
feature_importances_rf.sort_values(ascending=False).to_csv("RF_feature_importance.csv")

# Logistic Regression
feature_importances_lr.sort_values(ascending=False).to_csv("LR_feature_importance.csv")


### Multi-label confusion matrix ###

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix

# XGBoost prediction
y_pred_train_xgb = final_opt_xgb.predict(X_train_fs_mutinfo)

# Random Forest prediction
y_pred_train_rf = final_opt_rf.predict(X_train_fs_mutinfo)

# Logistic Regression prediction
y_pred_train_lr = final_opt_lr.predict(X_train_fs_mutinfo)

# Confusion matrix for XGBoost
confusion_mx_train_xgb = confusion_matrix(lc_y_train, y_pred_train_xgb)

# Confusion matrix for Random Forest
confusion_mx_train_rf = confusion_matrix(y_train, y_pred_train_rf)

# Confusion matrix for Logistic Regression
confusion_mx_train_lr = confusion_matrix(y_train, y_pred_train_lr)

# Creating dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.

# XGBoost
confusion_mx_train_xgb_df = pd.DataFrame(
    confusion_mx_train_xgb,
    index=["HER2+", "HR+", "Triple Neg"],
    columns=["HER2+", "HR+", "Triple Neg"],
)

# Random Forest
confusion_mx_train_rf_df = pd.DataFrame(
    confusion_mx_train_rf,
    index=["HER2+", "HR+", "Triple Neg"],
    columns=["HER2+", "HR+", "Triple Neg"],
)

# Logistic Regression
confusion_mx_train_lr_df = pd.DataFrame(
    confusion_mx_train_lr,
    index=["HER2+", "HR+", "Triple Neg"],
    columns=["HER2+", "HR+", "Triple Neg"],
)

# Plotting the confusion matrix - XGBoost
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mx_train_xgb_df, annot=True)
plt.title("XGBoost (training set)")
plt.ylabel("Actal Values")
plt.xlabel("Predicted Values")
plt.show()


# Plotting the confusion matrix - Random Forest
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mx_train_rf_df, annot=True)
plt.title("Random Forest (training set)")
plt.ylabel("Actal Values")
plt.xlabel("Predicted Values")
plt.show()

# Plotting the confusion matrix - Logistic Regression
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mx_train_lr_df, annot=True)
plt.title("Logistic Regression (training set)")
plt.ylabel("Actal Values")
plt.xlabel("Predicted Values")
plt.show()

### Multi-label F1-score


from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


# get f1 score (macro average)
f1_score_xgb = f1_score(lc_y_train, y_pred_train_xgb, average="macro")
f1_score_rf = f1_score(y_train, y_pred_train_rf, average="macro")
f1_score_lr = f1_score(y_train, y_pred_train_lr, average="macro")

precision_xgb = precision_score(lc_y_train, y_pred_train_xgb, average="macro")
precision_rf = precision_score(y_train, y_pred_train_rf, average="macro")
precision_lr = precision_score(y_train, y_pred_train_lr, average="macro")

recall_xgb = recall_score(lc_y_train, y_pred_train_xgb, average="macro")
recall_rf = recall_score(y_train, y_pred_train_rf, average="macro")
recall_lr = recall_score(y_train, y_pred_train_lr, average="macro")

accuracy_xgb = accuracy_score(lc_y_train, y_pred_train_xgb)
accuracy_rf = accuracy_score(y_train, y_pred_train_rf)
accuracy_lr = accuracy_score(y_train, y_pred_train_lr)

print("f1 score - XGBoost: ", f1_score_xgb)
print("f1 score - Random Forest: ", f1_score_rf)
print("f1 score - Logistic Regression: ", f1_score_lr)

print("precision - XGBoost: ", precision_xgb)
print("precision- Random Forest: ", precision_rf)
print("precision- Logistic Regression: ", precision_lr)

print("recall - XGBoost: ", recall_xgb)
print("recall - Random Forest: ", recall_rf)
print("recall - Logistic Regression: ", recall_lr)

print("accuracy - XGBoost: ", accuracy_xgb)
print("accuracy - Random Forest: ", accuracy_rf)
print("accuracy - Logistic Regression: ", accuracy_lr)


###====================================###
### Model evaluation (on test set) ###
###====================================###

# Make predictions on each model

# XGBoost
y_pred_test_xgb = final_opt_xgb.predict(X_test_fs_mutinfo)

# Random Forest
y_pred_test_rf = final_opt_rf.predict(X_test_fs_mutinfo)

# Logistic Regression
y_pred_test_lr = final_opt_lr.predict(X_test_fs_mutinfo)

# %%
# Make confustion matrix for each model on test set

# XGBoost
confusion_mx_test_xgb = confusion_matrix(lc_y_test, y_pred_test_xgb)

# Random Forest
confusion_mx_test_rf = confusion_matrix(y_test, y_pred_test_rf)

# Logistic Regression
confusion_mx_test_lr = confusion_matrix(y_test, y_pred_test_lr)


# Creating dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.

# XGBoost
confusion_mx_test_xgb_df = pd.DataFrame(
    confusion_mx_test_xgb,
    index=["HER2+", "HR+", "Triple Neg"],
    columns=["HER2+", "HR+", "Triple Neg"],
)

# Random Forest
confusion_mx_test_rf_df = pd.DataFrame(
    confusion_mx_test_rf,
    index=["HER2+", "HR+", "Triple Neg"],
    columns=["HER2+", "HR+", "Triple Neg"],
)

# Logistic Regression
confusion_mx_test_lr_df = pd.DataFrame(
    confusion_mx_test_lr,
    index=["HER2+", "HR+", "Triple Neg"],
    columns=["HER2+", "HR+", "Triple Neg"],
)

# Plotting the confusion matrix - XGBoost
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mx_test_xgb_df, annot=True)
plt.title("XGBoost (on test set)")
plt.ylabel("Actal values")
plt.xlabel("Predicted values")
plt.show()


# Plotting the confusion matrix - Random Forest
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mx_test_rf_df, annot=True)
plt.title("Random Forest (on test set)")
plt.ylabel("Actal Values")
plt.xlabel("Predicted Values")
plt.show()

# Plotting the confusion matrix - Logistic Regression
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mx_test_lr_df, annot=True)
plt.title("Logistic Regression (on test set)")
plt.ylabel("Actal Values")
plt.xlabel("Predicted Values")
plt.show()

# Final result - XGBoost

f1_xgb_test = f1_score(lc_y_test, y_pred_test_xgb, average="macro")
precision_xgb_test = precision_score(lc_y_test, y_pred_test_xgb, average="macro")
recall_xgb_test = recall_score(lc_y_test, y_pred_test_xgb, average="macro")
accuracy_xgb_test = accuracy_score(lc_y_test, y_pred_test_xgb)

print("f1-score: ", f1_xgb_test)
print("Precision: ", precision_xgb_test)
print("Recall: ", recall_xgb_test)
print("Accuracy: ", accuracy_xgb_test)

# Final result - Random Forest

f1_rf_test = f1_score(y_test, y_pred_test_rf, average="macro")
precision_rf_test = precision_score(y_test, y_pred_test_rf, average="macro")
recall_rf_test = recall_score(y_test, y_pred_test_rf, average="macro")
accuracy_rf_test = accuracy_score(y_test, y_pred_test_rf)

print("f1-score: ", f1_rf_test)
print("Precision: ", precision_rf_test)
print("Recall: ", recall_rf_test)
print("Accuracy: ", accuracy_rf_test)


# Final result - Logistic Regression

f1_lr_test = f1_score(y_test, y_pred_test_lr, average="macro")
precision_lr_test = precision_score(y_test, y_pred_test_lr, average="macro")
recall_lr_test = recall_score(y_test, y_pred_test_lr, average="macro")
accuracy_lr_test = accuracy_score(y_test, y_pred_test_lr)

print("f1-score: ", f1_lr_test)
print("Precision: ", precision_lr_test)
print("Recall: ", recall_lr_test)
print("Accuracy: ", accuracy_lr_test)


###=============================================================================###
### Mark the biomarker (Chromosome 17(35076296 - 35282086) ) for classification ###
###=============================================================================###

# Use Chr17 specific region as the only feature in the training and test sets
chr17_train = X_train[2184]
chr17_test = X_test[2184]


### GridsearchCV ###
# define mode
model_xgb_chr17 = XGBClassifier(random_state=42, num_class=3, objective="multi:softmax")

# define evaluation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space_xgb = dict()
space_xgb["max_depth"] = range(3, 10)
space_xgb["min_child_weight"] = range(1, 6)
space_xgb["gamma"] = [i / 10.0 for i in range(0, 5)]
# space_xgb['subsample'] = [i/10.0 for i in range(6,10)]
# space_xgb['colsample_bytree'] = [i/10.0 for i in range(6,10)]
space_xgb["reg_alpha"] = [1e-5, 1e-2, 0.1, 1, 100]

# define search_xgb
search_xgb_chr17 = GridSearchCV(
    model_xgb, space_xgb, scoring="accuracy", n_jobs=-1, cv=5
)

# execute search_xgb
result_xgb_chr17 = search_xgb_chr17.fit(chr17_train, lc_y_train)

# summarize result_xgb
print("Best Score: %s" % result_xgb_chr17.best_score_)
print("Best Hyperparameters: %s" % result_xgb_chr17.best_params_)

# print winning set of hyperparameters
from pprint import pprint

pprint(result_xgb_chr17.best_estimator_.get_params())

### Model evaluation on training set (chr17) ###
# (maybe this step is unnecessary; because what is the point we train again and use the same training set to make predictions?)

# XGBoost chr17
opt_xgb_chr17 = result_xgb_chr17.best_estimator_
final_opt_xgb_chr17 = opt_xgb_chr17.fit(chr17_train, lc_y_train)

# XGBoost chr17 prediction (training set)
y_pred_train_xgb_chr17 = final_opt_xgb_chr17.predict(chr17_train)

# Confusion matrix for XGBoost chr17
confusion_mx_train_xgb_chr17 = confusion_matrix(lc_y_train, y_pred_train_xgb_chr17)

# Creating dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
# XGBoost chr17
confusion_mx_train_xgb_chr17_df = pd.DataFrame(
    confusion_mx_train_xgb_chr17,
    index=["HER2+", "HR+", "Triple Neg"],
    columns=["HER2+", "HR+", "Triple Neg"],
)


# Plotting the confusion matrix - XGBoost chr17 (val set)
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mx_train_xgb_chr17_df, annot=True)
plt.title("XGBoost (feature chr17: 35076296 - 35282086) on training set)")
plt.ylabel("Actal Values")
plt.xlabel("Predicted Values")
plt.show()

# Final result - with only ffeature chr17

f1_chr17_train = f1_score(lc_y_train, y_pred_train_xgb_chr17, average="macro")
precision_chr17_train = precision_score(
    lc_y_train, y_pred_train_xgb_chr17, average="macro"
)
recall_chr17_train = recall_score(lc_y_train, y_pred_train_xgb_chr17, average="macro")
accuracy_chr17_train = accuracy_score(lc_y_train, y_pred_train_xgb_chr17)

print("f1-score: ", f1_chr17_train)
print("Precision: ", precision_chr17_train)
print("Recall: ", recall_chr17_train)
print("Accuracy: ", accuracy_chr17_train)

### Model evaluation on test set (chr17) ###
# (maybe this step is unnecessary; because what is the point we train again and use the same training set to make predictions?)

# XGBoost chr17
opt_xgb_chr17 = result_xgb_chr17.best_estimator_
final_opt_xgb_chr17 = opt_xgb_chr17.fit(chr17_train, lc_y_train)

# XGBoost chr17 prediction (test set)
y_pred_test_xgb_chr17 = final_opt_xgb_chr17.predict(chr17_test)

# Confusion matrix for XGBoost chr17
confusion_mx_test_xgb_chr17 = confusion_matrix(lc_y_test, y_pred_test_xgb_chr17)

# Creating dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
# XGBoost chr17
confusion_mx_test_xgb_chr17_df = pd.DataFrame(
    confusion_mx_test_xgb_chr17,
    index=["HER2+", "HR+", "Triple Neg"],
    columns=["HER2+", "HR+", "Triple Neg"],
)


# Plotting the confusion matrix - XGBoost chr17 (test set)
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mx_test_xgb_chr17_df, annot=True)
plt.title("XGBoost (feature chr17: 35076296 - 35282086) on test set)")
plt.ylabel("Actal testues")
plt.xlabel("Predicted testues")
plt.show()

# Final result - with only ffeature chr17

f1_chr17_test = f1_score(lc_y_test, y_pred_test_xgb_chr17, average="macro")
precision_chr17_test = precision_score(
    lc_y_test, y_pred_test_xgb_chr17, average="macro"
)
recall_chr17_test = recall_score(lc_y_test, y_pred_test_xgb_chr17, average="macro")
accuracy_chr17_test = accuracy_score(lc_y_test, y_pred_test_xgb_chr17)

print("f1-score: ", f1_chr17_test)
print("Precision: ", precision_chr17_test)
print("Recall: ", recall_chr17_test)
print("Accuracy: ", accuracy_chr17_test)
