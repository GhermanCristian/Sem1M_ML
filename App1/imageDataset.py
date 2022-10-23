from torch.utils.data import Dataset
from constants import Constants


class ImageDataset(Dataset):
    def __init__(self, imageList, imageClasses, transform):
        self.__images = []
        self.__labels = []
        self.__classes = list(set(imageClasses))
        self.__classToLabel = {className: imageName for imageName, className in enumerate(self.__classes)}

        self.__imageSize = Constants.IMAGE_SIZE  # square image
        self.__transformations = transform

        for image, imageClass in zip(imageList, imageClasses):
            self.__images.append(self.__transformations(image))
            self.__labels.append(self.__classToLabel[imageClass])

    def __getitem__(self, index):
        return self.__images[index], self.__labels[index]

    def __len__(self):
        return len(self.__images)
