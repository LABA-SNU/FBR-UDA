from models.dann import DANN
from models.dcoral import DeepCORAL
from models.ddc import DDCNet
from models.cdan import CDAN
from models.daln import DALN
from models.adamatch import AdamatchNet
from models.dadann import DADANN
from models.vanilaresnet import VanilaResNet


def get_model(name, backbone = 'resnet18', num_classes = 1000):
    if name == "dann_w_reshape":
        model = DANN(backbone = backbone, num_classes = num_classes, reshape = True)
        return model
    elif name == "dann":
        model = DANN(backbone = backbone, num_classes = num_classes, reshape = False)
        return model
    elif name == "dadann":
        model = DADANN(backbone = backbone, num_classes = num_classes, reshape = False)
        return model
    elif name == "dadann_w_reshape":
        model = DADANN(backbone = backbone, num_classes = num_classes, reshape = True)
        return model
    elif name == "dcoral":
        model = DeepCORAL(backbone = backbone, num_classes = num_classes)
        return model
    elif name == "ddc":
        model = DDCNet(backbone = backbone, num_classes = num_classes)
        return model
    elif name == "cdan":
        model = CDAN(backbone = backbone, num_classes = num_classes)
        return model
    elif name == "daln":
        model = DALN(backbone = backbone, num_classes = num_classes)
        return model
    elif name == "adamatch":
        model = AdamatchNet(backbone = backbone, num_classes = num_classes)
        return model
    elif name == "vanilaresnet":
        model = VanilaResNet(backbone = backbone, num_classes = num_classes)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))