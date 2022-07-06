import util

class Configurations:
    def __init__(self):
        config = util.LoadConfig('./config.json')

        self.FACENET_MODEL_PATH = config['FACENET_MODEL_PATH']
        self.SVM_MODEL_PATH = config['SVM_MODEL_PATH']
        self.NN_MODEL_PATH = config['NN_MODEL_PATH']
        self.TRAINING_SET_PATH = config['TRAINING_SET_PATH']
        self.VALIDATE_SET_PATH = config['VALIDATE_SET_PATH']
        self.COMPRESSED_FACE_IMAGE_DATASET_PATH = config['COMPRESSED_FACE_IMAGE_DATASET_PATH']
        self.COMPRESSED_FACE_EMBEDDING_PATH = config['COMPRESSED_FACE_EMBEDDING_PATH']
        self.FACE_CLASS_VIDEO_DIRECTORY = config['FACE_CLASS_VIDEO_DIRECTORY']
        self.VIDEO_SAMPLING_FREQUENCY = config['VIDEO_SAMPLING_FREQUENCY']
        self.FACES_CLASSIFICATION_DIRECTORY = config['FACES_CLASSIFICATION_DIRECTORY']
        self.FACES_CLASSIFICATION_OUTPUT = config['FACES_CLASSIFICATION_OUTPUT']
        self.FACES_CLASSES_COUNT = config['FACES_CLASSES_COUNT']