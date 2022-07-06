import util
import numpy
import sklearn.neighbors
import sklearn.metrics
import pickle
import Config

def EvaluateModel(model, validateX, validateY, trainY, config):
    p_at_k = numpy.zeros(len(validateX))

    for i in range(len(validateX)):
        x, gt_y = validateX[i], validateY[i]

        predicted_indexes, _ = util.ClassifyFaceEmbedding(model, x, config)
        predicted_labels = [trainY[id] for id in predicted_indexes]

        # validate the labels, if the predicted labels matches the gt label, mark the predicted index as 1, else 0
        # e.g. if the model predicts the input data to be closest to trained indexes [5, 11, 2, 7], if the
        # gt label of input data is 'John Liu', and the trained indexes corresponds to ['John Liu', 'John Liu', 'John Liu', 'Ben Affleck']
        # then the model has a 75% precision at k metric.
        validated_labels = [1 if label == gt_y else 0 for label in predicted_labels]
        p_at_k[i] = numpy.mean(validated_labels)

    return p_at_k.mean()

if __name__ == '__main__':
    config = Config.Configurations()

    # load training and validation set
    embedding_data = numpy.load(config.COMPRESSED_FACE_EMBEDDING_PATH)
    trainX, trainY, validateX, validateY = embedding_data['arr_0'], embedding_data['arr_1'], embedding_data['arr_2'], embedding_data['arr_3']
    print(f'Embeddings loaded, trainX: {trainX.shape}, trainY: {trainY.shape}, validateX: {validateX.shape}, validateY: {validateY.shape}')

    # fit nearest neighbor model for classification
    model = sklearn.neighbors.NearestNeighbors(n_neighbors=config.FACES_CLASSES_COUNT, algorithm='auto', metric='cosine')
    model.fit(trainX)

    # save the model to disk
    filename = config.NN_MODEL_PATH
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    # Evaluate the model for precision at k metric
    p_at_k = EvaluateModel(model, validateX, validateY, trainY, config)
    print(f'Precision at k: {p_at_k * 100}')