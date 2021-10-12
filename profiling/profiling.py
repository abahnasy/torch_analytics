import typing

import torch
import torch.nn as nn

from profiling.hooks import *

register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.

    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,
    nn.LayerNorm: count_ln,
    nn.InstanceNorm1d: count_in,
    nn.InstanceNorm2d: count_in,
    nn.InstanceNorm3d: count_in,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,

    nn.Sequential: zero_ops,

}


class Profiler():
    """
    """
    def __init__(self, model, custom_ops=None, verbose=True, ret_layer_info=False, report_missing=False) -> None:
        self.model = model
        self.custom_ops = custom_ops if custom_ops is not None else {}
        self.verbose = verbose
        self.ret_layer_info = ret_layer_info
        self.report_missing = report_missing

        self.handler_collection = {}
        self.types_collection = set()

        if self.report_missing:
            # overwrite `verbose` option when enable report_missing
            self.verbose = True

        

    
    def _add_hooks(self, m: nn.Module):
        m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))

        # for p in m.parameters():
        #     m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in self.custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = self.custom_ops[m_type]
            if m_type not in self.types_collection and self.verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in self.types_collection and self.verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in self.types_collection and self.report_missing:
                print("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if fn is not None:
            self.handler_collection[m] = (m.register_forward_hook(fn), m.register_forward_hook(count_parameters))
        self.types_collection.add(m_type)

    def _dfs_count(self, module: nn.Module, prefix="\t") -> typing.Tuple[int, int]:
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in self.handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = self._dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params

        return total_ops, total_params, ret_dict

    
    def profile(self, inputs: tuple):

        prev_training_status = self.model.training
        self.model.eval()
        self.model.apply(self._add_hooks)
        with torch.no_grad():
            self.model(*inputs)

        total_ops, total_params, ret_dict = self._dfs_count(self.model)
        
        # reset model to original status
        self.model.train(prev_training_status)

        for m, (op_handler, params_handler) in self.handler_collection.items():
            op_handler.remove()
            params_handler.remove()
            m._buffers.pop("total_ops")
            m._buffers.pop("total_params")

        if self.ret_layer_info:
            return total_ops, total_params, ret_dict
        return total_ops, total_params
