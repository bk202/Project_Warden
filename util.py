import mtcnn
import PIL.Image as Image
import numpy
import os
import json
from matplotlib import pyplot
import sklearn.svm

def LoadConfig(filePath):
    with open(filePath, "r") as f:
        return json.load(f)

def LoadImage(imagePath):
    image = Image.open(imagePath)
    image = image.convert('RGB')
    return numpy.asarray(image)

# apply MTCNN to detect all faces in image
def ExtractFacesFromImage(pixels, outputSize = (160, 160)):
    faceImages = list()
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(pixels)

    for result in results:
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]

        # resize face size to 160 x 160
        faceImage = Image.fromarray(face)
        faceImage = faceImage.resize(outputSize)
        # faceImage.save('./images/extraced_face.jpg')
        faceImages.append(numpy.asarray(faceImage))

    return faceImages

def ExtractFacesFromPath(path):
    faces = list()

    for fileName in os.listdir(path):
        imagePath = os.path.join(path, fileName)
        print(f'Processing image: {imagePath}')

        image = LoadImage(imagePath)

        # we can assume there is only 1 face in the training set
        faceImage = ExtractFacesFromImage(image)[0]
        faces.append(faceImage)

    return faces

def GenerateEmbedding(model, facePixels):
    """
    :param model: The loaded Keras FaceNet model
    :param facePixels: face pixels in 2d numpy array
    :return: the embedding of the face pixels
    """

    # Standardize the face pixels value for FaceNet input
    facePixels = facePixels.astype('float32')
    mean, std = facePixels.mean(), facePixels.std()
    facePixels = (facePixels - mean) / std

    # FaceNet performs prediction on batched images
    # insert a new dimension into facePixels to make a batch of 1 image
    samples = numpy.expand_dims(facePixels, axis=0)

    # make prediction to get embeddings
    embedding = model.predict(samples)[0]
    return embedding

def DisplayImageWithPrediction(image, title):
    """
    :param image: face image
    :return:
    """
    pyplot.imshow(image)
    pyplot.title(title)
    pyplot.show()

def TrainSVMFromEmbeddings(embeddingsPath):
    # load training and validation set
    data = numpy.load(embeddingsPath)
    trainX, trainY, validateX, validateY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(
        f'Embeddings loaded, trainX: {trainX.shape}, trainY: {trainY.shape}, validateX: {validateX.shape}, validateY: {validateY.shape}')

    # fit SVM for classification
    model = sklearn.svm.SVC(kernel='linear', probability=True)
    model.fit(trainX, trainY)
    return model


def ClassifyFaceEmbedding(nnModel, faceEmbedding, config):
    """
    :param svmModel: The NN model which performs classification based on face embedding
    :param faceEmbedding: the input face embedding
    :return: returns the data points stored in the NN clusters with distances lower than the set threshold
    """

    embedding = faceEmbedding.reshape(1, -1)

    '''
    since this is unsupervised learning, inds returns the indices of the nearest neighbors known
    in it's training set.

    We need to map each training point to it's specific class in order to interpret which class the prediction
    corresponds to.
    '''
    dists, indices = nnModel.kneighbors(X=embedding,
                                        n_neighbors=config.FACES_CLASSES_COUNT,
                                        return_distance=True)
    dists = dists[0]
    indices = indices[0]

    # filter out clusters that are below distance threshold of 0.6
    threshold = 0.5
    predictions = list()
    predictions_distances = list()

    for i in range(0, len(indices)):
        if (dists[i] > threshold):
            continue

        predictions.append(indices[i])
        predictions_distances.append([dists[i]])

    return predictions, predictions_distances

def Get_GroundTruth_labels(config):
    # load training and validation set
    embedding_data = numpy.load(config.COMPRESSED_FACE_EMBEDDING_PATH)
    trainX, trainY = embedding_data['arr_0'], embedding_data['arr_1']
    print(f'Embeddings loaded, trainX: {trainX.shape}, trainY: {trainY.shape}')

    return trainY