import torch
import torch.nn as nn
import torchvision
from opacus.validators import ModuleValidator

def load_model(model_name, num_classes, device, use_opacus=True):
    model_func = getattr(torchvision.models, model_name)
    model = model_func(num_classes=num_classes)
    if use_opacus:
        errors = ModuleValidator.validate(model, strict=False)
        print(errors)    
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
        print(errors)
    model.to(device)
    return model

