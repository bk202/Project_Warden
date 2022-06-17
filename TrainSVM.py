import util
import numpy
import sklearn.preprocessing
import sklearn.svm
import sklearn.metrics
import pickle

if __name__ == '__main__':
    config = util.LoadConfig('./config.json')

    # load training and validation set
    data = numpy.load(config['COMPRESSED_FACE_EMBEDDING_PATH'])
    trainX, trainY, validateX, validateY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(f'Embeddings loaded, trainX: {trainX.shape}, trainY: {trainY.shape}, validateX: {validateX.shape}, validateY: {validateY.shape}')

    # fit SVM for classification
    model = sklearn.svm.SVC(kernel='linear', probability=True)
    model.fit(trainX, trainY)

    # predict
    yhat_train = model.predict(trainX)
    yhat_validate = model.predict(validateX)

    # score
    score_train = sklearn.metrics.accuracy_score(trainY, yhat_train)
    score_validate = sklearn.metrics.accuracy_score(validateY, yhat_validate)
    print(f'Accuracy: train={score_train*100}, validate={score_validate*100}')

    # save the trained SVM model
    pickle.dump(model, open(config['SVM_MODEL_PATH'], 'wb'))

