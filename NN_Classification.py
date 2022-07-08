import util
import numpy
import pickle
import os
from keras.models import load_model
import Config
import shutil

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
        image_path = os.path.join(config.FACES_CLASSIFICATION_DIRECTORY, image)
        loaded_image = util.LoadImage(image_path)

        # assume there is only 1 face in image
        face_pixels = util.ExtractFacesFromImage(loaded_image)[0]
        face_embedding = util.GenerateEmbedding(faceNetModel, face_pixels)

        # predict the face based on random selection
        predicted_label, prediction_distance = ClassifyFaceEmbedding(nnModel, face_embedding, trainY, config)

        if config.VERBOSITY == 1:
            # plot for visual
            title = f'prediction: {predicted_label}, distance: {prediction_distance}'
            print(title)
            util.DisplayImageWithPrediction(face_pixels, title)

        # save image to records and remove image from classification path
        # append prediction result to output file
        with open(config.FACES_CLASSIFICATION_OUTPUT, 'a+') as f:
            line = f'{image} {predicted_label} {prediction_distance}\n'
            f.write(line)
            f.close()

        record_path = os.path.join(config.FACES_CLASSIFICATION_RECORD_DIRECTORY, image)
        shutil.copy(image_path, record_path)
        os.remove(image_path)