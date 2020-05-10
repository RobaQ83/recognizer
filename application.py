from tkinter import Tk

import torch
from torch.utils import model_zoo
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, model_urls

from gui import Application


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def main():
    network = resnet18(pretrained=True)
    network.eval()

    def callback(pixels):
        mapping = {
            5: "B",
            8: "A",
            6: "H",
        }
        with torch.no_grad():
            probabilities = network(pixels).softmax(dim=1)
            predicted_class = torch.max(probabilities, 1)[1]
            item = predicted_class.item()
            print(f"Most probable character: {item}")
            return mapping.get(item // 100, "?")

    root = Tk()
    application = Application(root, callback)
    application.start()


if __name__ == "__main__":
    main()
