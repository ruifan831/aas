from torchvision.models import vgg16
from torch import nn

class Backbone(nn.Module):

    def __init__(self,backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self,x):
        x = self.backbone(x)
        return x


def vggbackbone(pretrained = True):
    backbone = vgg16(pretrained)
    features = list(backbone.features)[:30]
    classifier = backbone.classifier

    classifier = list(classifier)
    del classifier[6]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    backbone = Backbone(nn.Sequential(*features))
    return backbone,classifier


