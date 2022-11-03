import os
from torchvision.datasets.folder import pil_loader
from constants import Constants


class ImageLoader:
    def __init__(self):
        self.images = []
        self.classes = []

    def loadImages(self, folderName):
        self.images, self.classes = [], []
        for _ in range(Constants.BUCKET_COUNT):
            self.images.append([])
            self.classes.append([])
        imageIndex = 0
        for imageName in os.listdir(folderName):
            self.images[imageIndex % Constants.BUCKET_COUNT].append(pil_loader(os.path.join(folderName, imageName)).convert('RGB'))
            self.classes[imageIndex % Constants.BUCKET_COUNT].append(imageName.startswith("COCO"))  # True => non album
            imageIndex += 1
