import util
import numpy
import sklearn.preprocessing
import sklearn.metrics
import pickle
from matplotlib import pyplot
from keras.models import load_model

if __name__ == '__main__':
    config = util.LoadConfig('./config.json')

    # load FaceNet model
    faceNetModel = load_model(config['FACENET_MODEL_PATH'])

    # load SVM model from disk
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

    selection = numpy.random.choice([i for i in range(validateX_faces.shape[0])])
    random_face_pixels = validateX_faces[selection]
    random_face_gt_name = validateY_faces[selection]
    random_face_embeddings = util.GenerateEmbedding(faceNetModel, random_face_pixels)

    # predict the face based on random selection
    samples = numpy.expand_dims(random_face_embeddings, axis=0)
    yhat_class = svmModel.predict(samples)[0]
    yhat_prob = svmModel.predict_proba(samples) * 100

    # convert prediction into class name
    yhat_class_name = out_encoder.inverse_transform([yhat_class])

    print(f'Predicted class name: {yhat_class_name} random_class_gt: {random_face_gt_name}, confidence: {yhat_prob}')

    # plot for visual
    pyplot.imshow(random_face_pixels)
    title = f'Predicted class name: {yhat_class_name} random_class_gt: {random_face_gt_name}, confidence: {yhat_prob}'
    pyplot.title(title)
    pyplot.show()