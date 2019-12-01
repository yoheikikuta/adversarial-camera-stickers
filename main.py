import argparse
import json

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from skimage import io
from torchvision import transforms


parser = argparse.ArgumentParser(description='Digital attack of adv. camera sticker.')

parser.add_argument('--imagenet_class_json',
                    help='Path to json file of Imagenet classes.',
                    default="./imagenet_class_index.json")

parser.add_argument('--target_img',
                    help='Path to image to be attacked.',
                    default="./image/1200px-STOP_sign.jpg")

parser.add_argument('--lr',
                    help='Initial learning rate.',
                    default=0.008)

parser.add_argument('--epoch',
                    help='Number of training epochs.',
                    default=200)

parser.add_argument('--lr_decay_interval',
                    help='Learning rate will be decayed after each this interval.',
                    default=50)

args = parser.parse_args()


class ImageDot(nn.Module):
    def __init__(self):
        super(ImageDot, self).__init__()
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.alpha = 0.5
        self.radius = 25.0
        self.beta = 2.0
        self.center = nn.Parameter(torch.tensor([
            [0.25, 0.25], [0.25, 0.5], [0.25, 0.75],
            [0.5, 0.25], [0.5, 0.5], [0.5, 0.75],
            [0.75, 0.25], [0.75, 0.5], [0.75, 0.75]]),
            requires_grad=True)
        self.color = nn.Parameter(torch.tensor([
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            requires_grad=True)

    def forward(self, x):
        _, _, height, width = x.shape
        blended = x
        for idx in range(self.center.shape[0]):
            mask = self._create_circle_mask(height, width,
                                            self.center[idx] * 255.0, self.beta)
            normalized_color = self._normalize_color(self.color[idx],
                                                     self.means, self.stds)
            blended = self._create_blended_img(blended, mask, normalized_color)
        return blended

    def _normalize_color(self, color, means, stds):
        return list(map(lambda x, m, s: (x - m) / s, color, means, stds))

    def _create_circle_mask(self, height, width, center, beta):
        hv, wv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        hv, wv = hv.type(torch.FloatTensor), wv.type(torch.FloatTensor)
        d = ((hv - center[0]) ** 2 + (wv - center[1]) ** 2) / self.radius ** 2
        return torch.exp(- d ** beta + 1e-10)

    def _create_blended_img(self, base, mask, color):
        alpha_tile = mask.expand(3, mask.shape[0], mask.shape[1])
        color_tile = torch.zeros_like(base)
        for c in range(3):
            color_tile[:, c, :, :] = color[c]
        return (1. - alpha_tile) * base + self.alpha * alpha_tile * color_tile


class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.image_dot = ImageDot()
        self.base_model = models.resnet50(pretrained=True).eval()
        self._freeze_pretrained_model()

    def _freeze_pretrained_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.image_dot(x)
        return self.base_model(x)


def predict_top_N(model: AttackModel, transformed_img: torch.Tensor,
                  N: int, idex2label: list, is_attacked=False) -> None:
    assert len(transformed_img.shape) == 3
    if is_attacked:
        pred = model(transformed_img.unsqueeze(0))
    else:
        pred = model.base_model(transformed_img.unsqueeze(0))
    pred = np.squeeze(pred.detach().numpy())

    for elem in np.argsort(pred)[-N:]:
        print(f"  class: {idx2label[elem]}, idx: {elem}, logit: {pred[elem]:.4f}")


def compute_loss(pred, true_label_idx=919, target_label_idx=595) -> torch.Tensor:
    # 919: street_sign
    # 595: harvester
    assert true_label_idx is not None
    true_label_contrib = F.nll_loss(pred, torch.tensor([true_label_idx]))
    target_label_contrib = F.nll_loss(pred, torch.tensor([target_label_idx]))
    if target_label_idx is None:
        return torch.mean(- true_label_contrib)  # non-targeted
    else:
        return torch.mean(- true_label_contrib + target_label_contrib)  # targeted


def load_class_json(img_path: str) -> list:
    with open(img_path) as json_file:
        class_idx = json.load(json_file)

    return [class_idx[str(k)][1] for k in range(len(class_idx))]


if __name__ == "__main__":
    torch.manual_seed(0)

    composed = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Image preparation
    img_array = io.imread(args.target_img)
    idx2label = load_class_json(args.imagenet_class_json)
    transformed_img = composed(PIL.Image.fromarray(img_array))

    model = AttackModel()

    print("Prediction result (before the attack):")
    predict_top_N(model, transformed_img, 10, idx2label)

    # Train model
    lr = args.lr
    loss_function = nn.NLLLoss()
    for epoch in range(args.epoch):
        if (epoch + 1) % args.lr_decay_interval == 0:
            lr /= 2.0
        model.zero_grad()
        pred = model(transformed_img.unsqueeze(0))
        loss = compute_loss(pred)
        loss.backward(retain_graph=True)  # type: ignore

        print(f"epoch: {epoch + 1}, loss: {loss.data:.4f}")

        for param in model.parameters():
            if param.requires_grad:
                param.data = torch.clamp(
                    param.data - param.grad.data * lr,  # type: ignore
                    min=0.0, max=1.0)

    print("Prediction result (after the attack):")
    predict_top_N(model, transformed_img, 10, idx2label, True)
