from Train import saved_model_knn

knn_from_pickle = pickle.loads(saved_model_knn)
knn_from_pickle.predict(testX)
