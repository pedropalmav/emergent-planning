import math
import torch
import torch.nn as nn


class ViTEmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = []

        self._register_hooks()

    def _register_hooks(self):
        for block in self.model.encoder.layers:
            block.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        tokens = output[:, 1:, :]
        B, N, C = tokens.shape
        H = W = int(math.sqrt(N))
        spatial_features = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        self.features.append(spatial_features)

    def forward(self, x):
        self.features = []
        return self.model(x)

    def get_features(self):
        return torch.cat(self.features, dim=1) if len(self.features) > 0 else None


if __name__ == "__main__":
    from vit import ViTBC

    base_model = ViTBC(
        image_size=8,
        patch_size=1,
        num_layers=5,
        num_heads=8,
        hidden_dim=64,
        mlp_dim=128,
        image_channels=7,
        num_classes=5,
    )

    model = ViTEmbeddingExtractor(base_model)

    model.eval()

    dummy_input = torch.randn(1, 7, 8, 8)
    with torch.no_grad():
        output = model(dummy_input)

    features = model.get_features()
    print(features.shape)
