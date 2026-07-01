import torch
import torch.nn as nn
from torchvision import models


# Fixed hyperparameter for L1 + VGG_r12.
# Change it here if you want another value.
LAMBDA_VGG_R12 = 0.01


def to_lpips_range(x):
    """
    LPIPS expects RGB images in [-1, 1].
    Your tensors are normalized to [0, 1],
    so this converts [0, 1] -> [-1, 1].
    """
    return x * 2.0 - 1.0


class VGGR12FeatureLoss(nn.Module):
    """
    VGG16 relu1_2 feature loss.

    Input tensors:
    - RGB
    - shape [N, 3, H, W]
    - range [0, 1]

    Important:
    - VGG is frozen.
    - VGG is cut at relu1_2 only, so it does not run full VGG16.
    """

    def __init__(self, normalize_features=True):
        super().__init__()

        self.normalize_features = normalize_features

        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            vgg = models.vgg16(weights=weights).features
        except AttributeError:
            vgg = models.vgg16(pretrained=True).features

        # Torchvision VGG16 features:
        # 0 conv1_1
        # 1 relu1_1
        # 2 conv1_2
        # 3 relu1_2
        #
        # vgg[:4] includes layers 0, 1, 2, 3.
        self.vgg = vgg[:4].eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _imagenet_norm(self, x):
        return (x - self.mean) / self.std

    @staticmethod
    def _normalize_feat(f, eps=1e-10):
        norm = torch.sqrt(torch.sum(f ** 2, dim=1, keepdim=True))
        return f / (norm + eps)

    def _extract(self, x):
        x = self._imagenet_norm(x)
        return self.vgg(x)

    def forward(self, output, target):
        # Keep images in valid range before VGG.
        output = output.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        output_feat = self._extract(output)

        # Target features do not need gradients.
        with torch.no_grad():
            target_feat = self._extract(target)

        if self.normalize_features:
            output_feat = self._normalize_feat(output_feat)
            target_feat = self._normalize_feat(target_feat)

        return torch.mean((output_feat - target_feat) ** 2)


class L1PlusVGGR12Loss(nn.Module):
    """
    Total loss:

        L_total = L1(output, target) + lambda * VGG_r12(output, target)

    where lambda is fixed by LAMBDA_VGG_R12.
    """

    def __init__(self, lambda_vgg=LAMBDA_VGG_R12):
        super().__init__()
        self.lambda_vgg = lambda_vgg
        self.l1 = nn.L1Loss()
        self.vgg_r12 = VGGR12FeatureLoss(normalize_features=True)

    def forward(self, output, target):
        # L1 uses raw output so out-of-range values still receive gradients.
        l1_loss = self.l1(output, target)

        # VGG loss internally clamps to [0, 1].
        vgg_loss = self.vgg_r12(output, target)

        return l1_loss + self.lambda_vgg * vgg_loss


def build_loss(loss_name):
    """
    Factory function for training losses.

    Available:
    - mse
    - l1_vgg_r12
    """
    loss_name = loss_name.lower()

    if loss_name == "mse":
        return nn.MSELoss()

    if loss_name == "l1_vgg_r12":
        return L1PlusVGGR12Loss()

    raise ValueError(
        f"Unknown loss: {loss_name}. "
        "Available losses: mse, l1_vgg_r12"
    )