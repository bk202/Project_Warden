import util
import numpy
import os
import Config

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

        faces = util.ExtractFacesFromPath(path)
        labels = [subDir for _ in range(len(faces))]

        # summarize progress
        print(f'> Loaded {len(faces)} faces for class {subDir}')

        x.extend(faces)
        y.extend(labels)

    return numpy.asarray(x), numpy.asarray(y)

def GenerateDataset(trainingSet, validateSet, outputFile):
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

if __name__ == '__main__':
    # config = util.LoadConfig('./config.json')
    config = Config.Configurations()
    GenerateDataset(trainingSet=config.TRAINING_SET_PATH,
                    validateSet=config.VALIDATE_SET_PATH,
                    outputFile=config.COMPRESSED_FACE_IMAGE_DATASET_PATH)