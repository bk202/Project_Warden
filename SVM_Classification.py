import util
import numpy
import sklearn.preprocessing
import sklearn.metrics
import pickle
import os
from keras.models import load_model

def ClassifyFaceEmbedding(svmModel, faceEmbedding):
    """
    :param svmModel: The SVM model which performs classification based on face embedding
    :param faceEmbedding: the input face embedding
    :return: returns the predicted class name and confidence (out of 100)
    """
    # predict the face based on random selection
    samples = numpy.expand_dims(faceEmbedding, axis=0)
    yhat_class = svmModel.predict(samples)[0]
    yhat_prob = svmModel.predict_proba(samples) * 100

    return yhat_class, yhat_prob

if __name__ == '__main__':
    config = util.LoadConfig('./config.json')

    # load FaceNet model
    faceNetModel = load_model(config['FACENET_MODEL_PATH'])

    # # load SVM model from disk
    svmModel = pickle.load(open(config['SVM_MODEL_PATH'], 'rb'))

    # load training and validation data set
    data = numpy.load(config['COMPRESSED_FACE_IMAGE_DATASET_PATH'])
    validateX_faces, validateY_faces = data['arr_2'], data['arr_3']
    print(f'Dataset loaded, validateX_faces: {validateX_faces.shape}, validateY: {validateY_faces.shape}')

    # convert labels into integers
    out_encoder = sklearn.preprocessing.LabelEncoder()
    out_encoder.fit(validateY_faces)
    validateY = out_encoder.transform(validateY_faces)
    print(f'Embeddings normalized, validateY: {validateY.shape}')

    for image in os.listdir(config['FACES_CLASSIFICATION_DIRECTORY']):
        imagePath = os.path.join(config['FACES_CLASSIFICATION_DIRECTORY'], image)

        print(f'image path: {imagePath}')

        image = util.LoadImage(imagePath)

        # assume there is only 1 face in image
        facePixels = util.ExtractFacesFromImage(image)[0]
        faceEmbedding = util.GenerateEmbedding(faceNetModel, facePixels)

        # predict the face based on random selection
        yhat_class, yhat_prob = ClassifyFaceEmbedding(svmModel, faceEmbedding)

        # plot for visual
        yhat_class_name = out_encoder.inverse_transform([yhat_class])

        print(f'prediction: {yhat_class_name}, confidencee: {yhat_prob}')
        confidence = max(yhat_prob[0])
        util.DisplayImageWithPrediction(facePixels, yhat_class_name, confidence)