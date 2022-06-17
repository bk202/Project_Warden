import util
import numpy
from keras.models import load_model
import sklearn.preprocessing

if __name__ == '__main__':
    config = util.LoadConfig('./config.json')

    # load training and validation set
    data = numpy.load(config['COMPRESSED_FACE_IMAGE_DATASET_PATH'])
    trainX, trainY, validateX, validateY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(f'Dataset loaded, trainX: {trainX.shape}, trainY: {trainY.shape}, validateX: {validateX.shape}, validateY: {validateY.shape}')

    model = load_model(config['FACENET_MODEL_PATH'])
    trainingEmbeddings = list()
    validateEmbeddings = list()

    for x in trainX:
        embedding = util.GenerateEmbedding(model, x)
        trainingEmbeddings.append(embedding)
    trainingEmbeddings = numpy.asarray(trainingEmbeddings)

    for x in validateX:
        embedding = util.GenerateEmbedding(model, x)
        validateEmbeddings.append(embedding)
    validateEmbeddings = numpy.asarray(validateEmbeddings)

    # normalize dataset with L2 normalizer to ensure all embeddings are measured in the same distance metric
    in_encoder = sklearn.preprocessing.Normalizer('l2')
    trainingEmbeddings = in_encoder.transform(trainingEmbeddings)
    validateEmbeddings = in_encoder.transform(validateEmbeddings)

    # convert labels into integers
    out_encoder = sklearn.preprocessing.LabelEncoder()
    out_encoder.fit(trainY)
    trainY = out_encoder.transform(trainY)
    validateY = out_encoder.transform(validateY)
    print(f'Embeddings normalized, trainX: {trainX.shape}, trainY: {trainY.shape}, validateX: {validateX.shape}, validateY: {validateY.shape}')

    print(f'trainingEmbeddings: {trainingEmbeddings.shape}, validateEmbeddings: {validateEmbeddings.shape}')
    numpy.savez_compressed(config['COMPRESSED_FACE_EMBEDDING_PATH'], trainingEmbeddings, trainY, validateEmbeddings, validateY)
