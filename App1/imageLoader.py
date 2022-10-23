import os
from torchvision.datasets.folder import pil_loader


def loadImages(folderName):
    images, classes = [], []
    for imageName in os.listdir(folderName):
        images.append(pil_loader(os.path.join(folderName, imageName)).convert('RGB'))
        classes.append(imageName.startswith("COCO"))
    return images, classes
