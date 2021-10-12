import types


import types

import torch
import torch.nn as nn
from torchvision.models import resnet18

from profiling.profiling import Profiler
from utils.si_prefixes import si_format

if __name__ == '__main__':

    model_complete = resnet18()
    model = resnet18()
    del model.avgpool
    del model.fc

    def forward(self, input_image):
        print(">>>>>", input_image.shape)
        x = self.conv1(input_image)
        x = self.bn1(x)
        f_0 = self.relu(x)
        x = self.maxpool(f_0)

        f_1 = self.layer1(x)
        f_2 = self.layer2(f_1)
        f_3 = self.layer3(f_2)
        f_4 = self.layer4(f_3)
        return f_4, f_3, f_2, f_1, f_0

    model.forward = types.MethodType(forward, model)

    print(model)
    input = torch.randn(1, 3, 224, 224)

    

    

    
    profiler1 = Profiler(model)
    macs1, params1 = profiler1.profile(inputs=(input,))

    profiler2 = Profiler(model_complete)
    macs2, params2 = profiler2.profile(inputs=(input,))

    print("macs and parameters are {} ({}) and {} ({})".format(macs1, si_format(macs1, precision=2),params1, si_format(params1, precision=2)))
    print("macs and parameters are {} ({}) and {} ({})".format(macs2, si_format(macs2, precision=2),params2, si_format(params2, precision=2)))
    
