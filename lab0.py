import numpy as np
import pylab as pl
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import tree, ensemble

import pandas as pd
from matplotlib import pyplot as plt
def plot_results_with_hyperplane(clf, clf_name, df, plt_nmbr):
    x_min, x_max = df.x.min() - .5, df.x.max() + .5
    y_min, y_max = df.y.min() - .5, df.y.max() + .5
    # step between points. i.e. [0, 0.02, 0.04, ...]
    step = .02
    # to plot the boundary, we're going to create a matrix of every possible point
    # then label each point as a wolf or cow using our classifier
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # this gets our predictions back into a matrix
    Z = Z.reshape(xx.shape)
    # create a subplot (we're going to have more than 1 plot on a given image)
    pl.subplot(2, 3, plt_nmbr)
    # plot the boundaries
    pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)
    # plot the wolves and cows
    for animal in df.animal.unique():
        pl.scatter(df[df.animal==animal].x,
                   df[df.animal==animal].y,
                   marker=animal, s=70,
                   label="cows" if animal=="x" else "wolves",
                   color='black')
    pl.title(clf_name, fontsize=20)

data = open("cows_and_wolves.txt").read()
data = [row.split('\t') for row in data.strip().split('\n')]

animals = []
for y, row in enumerate(data):
    for x, item in enumerate(row):
        # x's are cows, o's are wolves
        if item in ['o', 'x']:
            animals.append([x, y, item])

df = pd.DataFrame(animals, columns=["x", "y", "animal"])
df['animal_type'] = df.animal.apply(lambda x: 0 if x=="x" else 1)

# train using the x and y position coordiantes
train_cols = ["x", "y"]

clfs = {
    "SVM": svm.SVC(),
    "Logistic Regression" : linear_model.LogisticRegression(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": ensemble.RandomForestClassifier(random_state=0),
    "K-Nearest Neighbors Classifier": KNeighborsClassifier(n_neighbors=3), 
    "Gaussian Naive Bayes": GaussianNB(),
}
plt.figure(figsize=(20,12))
plt_nmbr = 1
for clf_name, clf in clfs.items():
    clf.fit(df[train_cols], df.animal_type)
    plot_results_with_hyperplane(clf, clf_name, df, plt_nmbr)
    plt_nmbr += 1
pl.show()