import util
import cv2
import os
import numpy
import shutil
import Config

def GetClassName(fileName):
    return fileName.split('.')[0]

if __name__ == '__main__':
    config = Config.Configurations()
    samplingFrequency = config.VIDEO_SAMPLING_FREQUENCY
    videos = os.listdir(config.FACE_CLASS_VIDEO_DIRECTORY)
    sampleFrames = list()

    for video in videos:
        className = GetClassName(video)
        videoPath = os.path.join(config.FACE_CLASS_VIDEO_DIRECTORY, video)
        samplingCount = 1

        cap = cv2.VideoCapture(videoPath, 0)
        while (cap.isOpened()):
            ret, frame = cap.read()

            if (ret):
                # save sampling frame if count reaches sampling frequency
                if (samplingCount == samplingFrequency):
                    sampleFrames.append(frame)
                    samplingCount = 1
                else:
                    samplingCount += 1
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f'class name: {className}, total sampled frames: {len(sampleFrames)}')

        # shuffle sampled frames and split in half
        numpy.random.shuffle(sampleFrames)
        mid = len(sampleFrames) // 2
        trainingSet = sampleFrames[:mid]
        validateSet = sampleFrames[mid:]

        trainingPath = os.path.join(config.TRAINING_SET_PATH, className)
        if (os.path.exists(trainingPath)):
            shutil.rmtree(trainingPath)
        os.mkdir(trainingPath)

        index = 0
        for img in trainingSet:
            img_name = f'{className}_{index}.jpg'
            img_path = os.path.join(trainingPath, img_name)
            cv2.imwrite(img_path, img)
            index += 1

        validatePath = os.path.join(config.VALIDATE_SET_PATH, className)
        if (os.path.exists(validatePath)):
            shutil.rmtree(validatePath)
        os.mkdir(validatePath)

        index = 0
        for img in validateSet:
            img_name = f'{className}_{index}.jpg'
            img_path = os.path.join(validatePath, img_name)
            cv2.imwrite(img_path, img)
            index += 1
