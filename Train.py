# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import os
import pickle


# FUNCTION FOR CHANGING COLORGRADE OF IMAGE
def extract_color_stats(image):
    # split the input image into its respective RGB color channels
    # and then create a feature vector with 6 values: the mean and
    # standard deviation for each of the 3 channels, respectively
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]
    # return our set of features
    return features


# grab all image paths in the input dataset directory
dataset = "Dataset"
imagePaths = paths.list_images(dataset)
data = []
labels = []

# test data setup
# testset = "Test"
# testImagePaths = paths.list_images(testset)
# testData = []
# testLabels = []

# loop over our input images
for imagePath in imagePaths:
    # load the input image from disk, compute color channel
    # statistics, and then update our data list
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)
    # extract the class label from the file path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# for test in testImagePaths:
#     image = Image.open(test)
#     features = extract_color_stats(image)
#     testData.append(features)
#     testLabel = test.split(os.path.sep)[-2]
#     testLabels.append(testLabel)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

# Encode the test labels
# testLabels = le.fit_transform(testLabels)
# (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(testData, testLabels, test_size=0.25)

# train the model
knn = KNN(n_neighbors=3)
rf = RFC(n_estimators=100)

# Show the accuracy of the models
# knn.fit(trainX, trainY)
# predictions = knn.predict(testX)
# print("REPORT FOR KNN")
# print(classification_report(testY, predictions, target_names=le.classes_))

filename = "knn_model.sav"
loaded_model = pickle.load(open(filename, "wb"))
predict = loaded_model.predict(testX)
print(classification_report(testY, predict, target_names=le.classes_))
# rf.fit(trainX, trainY)
# predictions = rf.predict(testX)
# print("REPORT FOR RF")
# print(classification_report(testY, predictions, target_names=le.classes_))


# Save the trained model
# pickle.dump(knn, open(filename, "wb"))
# saved_model_knn = pickle.dumps(knn)
# saved_model_rf = pickle.dumps(rf)

# loaded_model = pickle.load(open(filename, "rb"))
# result = loaded_model.predict(Xtest)
# print(classification_report(Ytest, result, target_names=le.classes_))
# print(result)
