import torch
from torch.autograd import Variable
import PIL.Image
from torchvision.transforms import transforms
from simpleNet import SimpleNet


def getImage(imageName):
    transformations = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()])
    return Variable(transformations(PIL.Image.open(imageName)).float(),
                    requires_grad=True).unsqueeze(0)


def run():
    model = SimpleNet(2)
    model.load_state_dict(torch.load("model-name-without-extension"))
    model.eval()
    output = model(getImage('path-to-image'))
    _, prediction = torch.max(output.data, 1)
    if prediction.item():
        print("Not an album")
    else:
        print("Album")
