import util
import numpy
import sklearn.preprocessing
import sklearn.metrics
import pickle
import os
from keras.models import load_model
import Config

def Get_GroundTruth_labels(config):
    # load training and validation set
    embedding_data = numpy.load(config.COMPRESSED_FACE_EMBEDDING_PATH)
    trainX, trainY = embedding_data['arr_0'], embedding_data['arr_1']
    print(f'Embeddings loaded, trainX: {trainX.shape}, trainY: {trainY.shape}')

    return trainY

def ClassifyFaceEmbedding(nnModel, faceEmbedding, trainY, config):
    """
    :param svmModel: The NN model which performs classification based on face embedding
    :param faceEmbedding: the input face embedding
    :return: returns the predicted class name and confidence (out of 100)
    """

    predicted_indexes, prediction_distances = util.ClassifyFaceEmbedding(nnModel, faceEmbedding, config)
    predicted_labels = [trainY[id] for id in predicted_indexes]

    if (len(predicted_labels) == 0):
        return 'unknown', float('INF')
    else:
        print(f'Predictions: {predicted_labels}, distances: {prediction_distances}')
        return predicted_labels[0], prediction_distances[0]

if __name__ == '__main__':
    config = Config.Configurations()

    # load FaceNet model
    faceNetModel = load_model(config.FACENET_MODEL_PATH)

    # load ground truth labels
    trainY = Get_GroundTruth_labels(config)

    # # load SVM model from disk
    nnModel = pickle.load(open(config.NN_MODEL_PATH, 'rb'))

    for image in os.listdir(config.FACES_CLASSIFICATION_DIRECTORY):
        imagePath = os.path.join(config.FACES_CLASSIFICATION_DIRECTORY, image)

        print(f'image path: {imagePath}')

        image = util.LoadImage(imagePath)

        # assume there is only 1 face in image
        facePixels = util.ExtractFacesFromImage(image)[0]
        faceEmbedding = util.GenerateEmbedding(faceNetModel, facePixels)

        # predict the face based on random selection
        predicted_label, prediction_distance = ClassifyFaceEmbedding(nnModel, faceEmbedding, trainY, config)

        # plot for visual
        title = f'prediction: {predicted_label}, distance: {prediction_distance}'
        print(title)
        util.DisplayImageWithPrediction(facePixels, title)