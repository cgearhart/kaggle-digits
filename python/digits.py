
import pandas as pd
import numpy as np

from sklearn import cross_validation
from sklearn.decomposition import FastICA, RandomizedPCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# small ICA 56, 2, 1000, 1 -- .78 +-.08 509.5s

# User settings
DATASET = "full"  # ["small", "full"]
FILTERNAME = "PCA"  # ["identity", "PCA", "ICA"]
FILTER_OPTS = {"n_components": 56, "whiten": True}
CROSS_VALIDATION_ROUNDS = 5
# CLASSIFIER = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
#                                 n_estimators=1000,
#                                 learning_rate=1
#                                 )
CLASSIFIER = RandomForestClassifier(n_estimators=500)

# Constants
OUTFILE = "./results/results.csv"
TRAINFILES = {
              "small": "./data/train_small.csv",
              "full": "./data/train.csv"
             }
TESTFILES = {
             "small": "./data/test_small.csv",
             "full": "./data/test.csv"
            }
FILTERS = {
           "identity": None,
           "PCA": RandomizedPCA,
           "ICA": FastICA
          }


if __name__ == "__main__":

    train_infile = TRAINFILES[DATASET]
    test_infile = TESTFILES[DATASET]
    dataFilter = FILTERS[FILTERNAME]
    if dataFilter:
        dataFilter = dataFilter(**FILTER_OPTS)

    logfmt = "Dataset: {}\nFilter: {!s}\nClassifier: {!s}"
    print logfmt.format(DATASET, dataFilter, CLASSIFIER)

    print "Reading training data..."
    training_df = pd.read_csv(train_infile, delimiter=",")
    x = training_df.iloc[:, 1:].as_matrix().astype(np.double)
    y = training_df['label'].as_matrix()

    print "Reading test data..."
    test_df = pd.read_csv(test_infile, delimiter=",")
    x_t = test_df.as_matrix().astype(np.double)

    print "Preprocessing data..."
    if dataFilter:
        x = dataFilter.fit_transform(x)
        x_t = dataFilter.transform(x_t)

    print "Training set size: {}\nTest set size: {}".format(x.shape[0],
                                                            x_t.shape[0])

    print "Cross validation testing..."
    print "CV Rounds: {}".format(CROSS_VALIDATION_ROUNDS)
    scores = cross_validation.cross_val_score(CLASSIFIER, x, y,
                                              cv=CROSS_VALIDATION_ROUNDS)
    print "CV Accuracy: {:0.4f} (+/- {:0.4f})".format(scores.mean(),
                                                      scores.std() * 2)

    print "Training full model..."
    CLASSIFIER.fit(x, y)

    y_hat = CLASSIFIER.predict(x)
    cm = confusion_matrix(y, y_hat)
    df = pd.DataFrame(cm)
    print "Confusion Matrix:\n{!s}".format(df)

    print "Making predictions..."
    final_predictions = pd.Series(CLASSIFIER.predict(x_t), name="Label")
    final_predictions.index += 1  # adjust the index to start at 1
    final_predictions.to_csv(OUTFILE, index_label="ImageId", header=True)

    print "Done."
