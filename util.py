
# pre-process image
import mtcnn
import PIL.Image as Image
import numpy
import os
from keras.models import load_model

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

def LoadDataset(directory):
    """
    :param directory: string which indicates the dataset directory
    :return: x: list of faces converted to pixel, y: list of class label in string
    """
    x, y = list(), list()

    for subDir in os.listdir(directory):
        path = os.path.join(directory, subDir)

        if (not os.path.isdir(path)):
            continue

        faces = ExtractFacesFromPath(path)
        labels = [subDir for _ in range(len(faces))]

        # summarize progress
        print(f'> Loaded {len(faces)} faces for class {subDir}')

        x.extend(faces)
        y.extend(labels)

    return numpy.asarray(x), numpy.asarray(y)

def GenerateDataset(
        trainingSet = './5-celebrity-faces-dataset/train/',
        validateSet = './5-celebrity-faces-dataset/val/',
        outputFile = '5-celebrity-faces-dataset.npz'
):
    """
    load training set and test set and save into .npz file
    :return: None
    """
    trainX, trainY = LoadDataset(trainingSet)
    print(f'trainX: {trainX.shape}, trainY: {trainY.shape}')

    validateX, validateY = LoadDataset(validateSet)
    print(f'validateX: {validateX.shape}, validateY: {validateY.shape}')
    numpy.savez_compressed(outputFile, trainX, trainY, validateX, validateY)

    return None

def GenerateEmbeddings(modelPath = './keras-facenet/model/facenet_keras.h5'):
    model = load_model('./keras-facenet/model/facenet_keras.h5')
    print(model.inputs)
    print(model.outputs)