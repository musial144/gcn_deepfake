import open_clip
import torch
import torch
import torch.nn.functional as F

"""
Klasa obsługująca ekstrakchę cech z patchy. Wykorzystuje model CLIP z biblioteki open_clip. 
PARAMETRY:
    model_name  - nazwa modelu CLIP do wykorzystania. Konfigurowalne w pliku config.yaml
    pretrained  - czy wykorzystać model wstępnie wytrenowany (pretrained='openai') 
                  czy losowo zainicjalizowany (pretrained=None). Konfigurowalne w pliku configu.yaml
PROCEDURY:
    _scale_patch()     - skaluje patch do rozmiaru wymaganego przez model CLIP (224x224). Rozmiar konfigurowalny w pliku config.yaml
    extract_features() - ekstrahuje cechy z patcha za pomocą modelu CLIP. Nie klasyfikuje, zwraca jedynie cechy
"""
class CLIPExtractor(torch.nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained = 'openai'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained, device=self.device
            )
        
        self.model.eval().to(self.device)
        self.out_dim = self.model.visual.output_dim
        self.img_size = 224 # 224 bo CLIP przyjmuje taki rozmiar (224x224 - tensor.size >= 50)

    def _scale_patch(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        if patch_tensor.shape[-2:] != (self.img_size, self.img_size):
            patch_tensor = F.interpolate(patch_tensor, size = (self.img_size, self.img_size))

        return patch_tensor
        

    @torch.no_grad
    def extract_features(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        patch_tensor = self._scale_patch(patch_tensor)

        # feats = self.model.encode_image(patch_tensor, proj = False)
        feats = self.model.visual(patch_tensor)
        # feats może być [B, tokens, D] albo [B, D] zależnie od modelu
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        # feats /= feats.norm(dim=-1, keepdim=True)
        
        return feats
