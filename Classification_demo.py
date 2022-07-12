import numpy

import util
import pickle
from keras.models import load_model
import Config
import cv2
import mtcnn
import PIL.Image as Image

def ClassifyFaceEmbedding(nnModel, faceEmbedding, trainY, config):
    """
    :param svmModel: The NN model which performs classification based on face embedding
    :param faceEmbedding: the input face embedding
    :return: returns the predicted class name and confidence (out of 100)
    """

    predicted_indexes, prediction_distances = util.ClassifyFaceEmbedding(nnModel, faceEmbedding, config)
    predicted_labels = [trainY[id] for id in predicted_indexes]

    if (len(predicted_labels) == 0):
        return 'unknown', float('INF')
    else:
        print(f'Predictions: {predicted_labels}, distances: {prediction_distances}')
        return predicted_labels[0], prediction_distances[0]

if __name__ == '__main__':
    config = Config.Configurations()

    # load FaceNet model
    faceNetModel = load_model(config.FACENET_MODEL_PATH)

    # load ground truth labels
    trainY = util.Get_GroundTruth_labels(config)

    # # load SVM model from disk
    nnModel = pickle.load(open(config.NN_MODEL_PATH, 'rb'))

    # load video
    video_path = config.DEMO_INPUT_VIDEO_PATH

    cap = cv2.VideoCapture(video_path, 0)

    frames = list()
    detector = mtcnn.MTCNN()
    samplingFrequency = 3
    samplingCount = 0

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            # reduce the number of frames by applying sampling
            if (samplingCount == samplingFrequency):
                frames.append(frame)
                samplingCount = 0
            else:
                samplingCount += 1
        else:
            break

    print(f'frames: {len(frames)}')
    for frame in frames:
        face_bounding_box = detector.detect_faces(frame)[0]
        x1, y1, width, height = face_bounding_box['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # perform classification
        face_pixels = frame[y1:y2, x1:x2]
        face_pixels = Image.fromarray(face_pixels)
        face_pixels = face_pixels.resize((160, 160))

        face_embedding = util.GenerateEmbedding(faceNetModel, numpy.asarray(face_pixels))

        # predict the face based on random selection
        predicted_label, prediction_distance = ClassifyFaceEmbedding(nnModel, face_embedding, trainY, config)

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        text = f'{predicted_label}, {prediction_distance}'
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
