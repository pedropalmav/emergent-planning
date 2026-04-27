import math
import torch
import torch.nn as nn


class ResNetEmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = []

        self._register_hooks()

    def _register_hooks(self):
        for block in self.model.core:
            block.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features.append(output)

    def forward(self, x):
        self.features = []
        return self.model(x)

    def get_features(self):
        return torch.cat(self.features, dim=1) if len(self.features) > 0 else None


if __name__ == "__main__":
    from resnet import ResNetBC

    base_model = ResNetBC(
        image_size=8,
        kernel_size=3,
        num_layers=24,
        hidden_dim=32,
        mlp_dim=128,
        num_classes=5,
        image_channels=7,
    )

    model = ResNetEmbeddingExtractor(base_model)

    model.eval()

    dummy_input = torch.randn(1, 7, 8, 8)
    with torch.no_grad():
        output = model(dummy_input)

    features = model.get_features()
    print(features.shape)
