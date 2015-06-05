
from sklearn import cross_validation

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# hyperparameters
D = 2
N = 1000
L = 1

classifier = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=D),
            n_estimators=N,
            learning_rate=L
            )


def cv(train, labels):
    print "Cross validation testing..."
    scores = cross_validation.cross_val_score(classifier, train, labels, cv=5)
    print "Accuracy: {:0.2f} (+/- {:0.2f})".format(scores.mean(),
                                                   scores.std() * 2)


def predict(train, labels, test):
    print "Training final model..."
    classifier.fit(train, labels)

    print "Making predictions..."
    final_predictions = pd.Series(classifier.predict(x_t), name="Label")
    final_predictions.index += 1  # adjust the index to start at 1
    final_predictions.to_csv(outfile, index_label="ImageId", header=True)

