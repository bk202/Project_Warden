import mtcnn
import PIL.Image as Image
import numpy
import os
import json
from matplotlib import pyplot

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

def DisplayImageWithPrediction(image, prediction, probability):
    """
    :param image: face image
    :param prediction: the predicted class name
    :param probability: confidence of prediction
    :return:
    """
    pyplot.imshow(image)
    title = f'Predicted class name: {prediction}, confidence: {probability}%'
    pyplot.title(title)
    pyplot.show()
