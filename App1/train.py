import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import transforms
from imageLoader import ImageLoader
from constants import Constants
from imageDataset import ImageDataset
from simpleNet import SimpleNet
import numpy as np


class Train:
    def __init__(self):
        self.__model = SimpleNet(2)
        self.__optimizer = Adam(self.__model.parameters(), lr=0.001, weight_decay=0.0001)
        self.__lossFunction = nn.CrossEntropyLoss()
        self.__imageLoader = ImageLoader()
        self.__imageLoader.loadImages("Datasets/Data")

    def __getTrainLoader(self, testingBucketIndex):
        trainSetImages, trainSetClasses = [], []
        for i in range(Constants.BUCKET_COUNT):
            if i != testingBucketIndex:
                trainSetImages.extend(self.__imageLoader.images[i])
                trainSetClasses.extend(self.__imageLoader.classes[i])
        trainTransformations = transforms.Compose([
            transforms.Resize((Constants.IMAGE_SIZE, Constants.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainSet = ImageDataset(trainSetImages, trainSetClasses, trainTransformations)
        return DataLoader(trainSet, batch_size=Constants.BATCH_SIZE, shuffle=True, num_workers=4)

    def __getTestLoader(self, testingBucketIndex):
        testSetImages = self.__imageLoader.images[testingBucketIndex]
        testSetClasses = self.__imageLoader.classes[testingBucketIndex]
        testTransformations = transforms.Compose([
            transforms.Resize((Constants.IMAGE_SIZE, Constants.IMAGE_SIZE)),
            transforms.CenterCrop(Constants.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        testSet = ImageDataset(testSetImages, testSetClasses, testTransformations)
        return DataLoader(testSet, batch_size=Constants.BATCH_SIZE, shuffle=False, num_workers=4)

    def __saveModels(self, epoch):
        torch.save(self.__model.state_dict(), "myModel_{}.model".format(epoch))
        print("Checkpoint saved")

    def __test(self, testingBucketIndex):
        self.__model.eval()
        testAccuracy = 0.0
        testLoader = self.__getTestLoader(testingBucketIndex)
        for i, (images, labels) in enumerate(testLoader):
            outputs = self.__model(images)
            _, prediction = torch.max(outputs.data, 1)
            testAccuracy += torch.sum(torch.eq(prediction, labels.data))

        testAccuracy = testAccuracy / Constants.TEST_IMAGE_COUNT
        return testAccuracy

    def __train(self, epochCount):
        bestTestAccuracyMean = 0.0

        for epoch in range(epochCount):
            trainAccuracyList = []
            trainLossList = []
            testAccuracyList = []

            for testingBucketIndex in range(Constants.BUCKET_COUNT):
                trainLoader = self.__getTrainLoader(testingBucketIndex)
                self.__model.train()
                trainAccuracy = 0.0
                trainLoss = 0.0
                for i, (images, labels) in enumerate(trainLoader):
                    self.__optimizer.zero_grad()  # Clear all accumulated gradients
                    outputs = self.__model(images)
                    loss = self.__lossFunction(outputs, labels)
                    loss.backward()
                    self.__optimizer.step()

                    trainLoss += loss.cpu().data.item() * images.size(0)
                    _, prediction = torch.max(outputs.data, 1)

                    trainAccuracy += torch.sum(prediction == labels.data)

                trainAccuracy = trainAccuracy / Constants.TRAIN_IMAGE_COUNT
                trainLoss = trainLoss / Constants.TRAIN_IMAGE_COUNT
                testAccuracy = self.__test(testingBucketIndex)
                print("Bucket {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(testingBucketIndex, trainAccuracy, trainLoss, testAccuracy))

                trainAccuracyList.append(trainAccuracy)
                trainLossList.append(trainLoss)
                testAccuracyList.append(testAccuracy)

            trainAccuracyMean = np.mean(trainAccuracyList)
            trainLossMean = np.mean(trainLossList)
            testAccuracyMean = np.mean(testAccuracyList)
            testAccuracyStd = np.std(testAccuracyList)
            CILow = testAccuracyMean - 1.96 * testAccuracyStd / np.sqrt(Constants.BUCKET_COUNT)
            CIHigh = testAccuracyMean + 1.96 * testAccuracyStd / np.sqrt(Constants.BUCKET_COUNT)

            if bestTestAccuracyMean < testAccuracyMean:
                bestTestAccuracyMean = testAccuracyMean
                self.__saveModels(epoch)

            # Print the metrics
            print("Epoch {}, Train Accuracy mean: {} , TrainLoss mean: {} , Test Accuracy mean: {}, CI: [{}, {}]".format(epoch, trainAccuracyMean, trainLossMean, testAccuracyMean, CILow, CIHigh))

    def runProgram(self):
        torch.cuda.empty_cache()
        self.__train(Constants.EPOCH_COUNT)
