import sys
import time
import numpy as np
import pickle
import os
import gzip
import numpy as np
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def load_mnist(path, kind='train'):
    '''

    load mnist fashion dataset.

    '''

    labels_path = os.path.join(
        path,
        '{}-labels-idx1-ubyte.gz'.format(kind)
    )
    images_path = os.path.join(
        path,
        '{}-images-idx3-ubyte.gz'.format(kind)
    )

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(
            lbpath.read(),
            dtype=np.uint8,
            offset=8
        )

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(),
            dtype=np.uint8,
            offset=16
        ).reshape(len(labels), 784)

    return images, labels

def fit(train):
    '''

    Generate svm model.

    '''

    # split data
    X = train[0]
    y = train[1]

    # generate model
    clf = svm.SVC(gamma=0.1, kernel='poly')
    clf.fit(X, y)

    return clf

def predict(clf, test):
    '''

    Use SVM model to predict.

    '''

    return clf.predict(test[0])

def accuracy(clf, test, y_pred):
    '''

    Determine accuracy and confusion matrix.

    '''

    accuracy_classifier = clf.score(test[0], test[1])
    accuracy_prediction = accuracy_score(test[1], y_pred)
    confusion = confusion_matrix(test[1], y_pred)

    print('\nSVM Trained Classifier Accuracy: {}'.format(accuracy_classifier))
    print('Predicted Values: {}'.format(y_pred))
    print('Accuracy of Classifier on Validation Images: {}'.format(accuracy_prediction))
    print('Confusion Matrix: {}\n'.format(confusion))

    # confusion matrix plot
    plt.matshow(confusion)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # predicting test images
    predicted_labels = clf.predict(test[0])

    # classifier accuracy: using test data
    accuracy_test = accuracy_score(test[1], predicted_labels)

    # confusion matrix
    confusion_test = confusion_matrix(test[1], predicted_labels)

    print('Predicted Labels for Test Images: {}'.format(predicted_labels))
    print('Accuracy of Classifier on Test Images: '.format(accuracy_test))
    print('Confusion Matrix for Test Data: \n'.format(confusion_test))

    # confusion matrix: using test data
    plt.matshow(confusion_test)
    plt.title('Confusion Matrix for Test Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return predicted_labels

def viz_svm(lbl, test, predicted_labels):
    '''

    Visualize predictions against original values.

    '''

    arr = lbls.values()

    # original vs predicted labels: test images
    a = np.random.randint(1,40,15)
    for i in a:
        two_d = (np.reshape(test[i], (28, 28)) * 255).astype(np.uint8)
        plt.title('Original Label: {0}  Predicted Label: {1}'.format(
            arr[test[i]],
            arr[predicted_labels[i]]
        ))
        plt.imshow(two_d, interpolation='nearest')
        plt.show()