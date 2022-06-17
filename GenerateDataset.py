import util
import numpy

def GenerateDataset(trainingSet, validateSet, outputFile):
    """
    load training set and test set and save into .npz file
    :return: None
    """
    trainX, trainY = util.LoadDataset(trainingSet)
    print(f'trainX: {trainX.shape}, trainY: {trainY.shape}')

    validateX, validateY = util.LoadDataset(validateSet)
    print(f'validateX: {validateX.shape}, validateY: {validateY.shape}')
    numpy.savez_compressed(outputFile, trainX, trainY, validateX, validateY)

    return None

if __name__ == '__main__':
    config = util.LoadConfig('./config.json')
    GenerateDataset(trainingSet=config['TRAINING_SET_PATH'],
                    validateSet=config['VALIDATE_SET_PATH'],
                    outputFile=config['COMPRESSED_FACE_IMAGE_DATASET_PATH'])