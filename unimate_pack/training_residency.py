"""ComfyUI-selected full model residency for node-owned training state."""
from contextlib import contextmanager

from .training_transforms import _check


@contextmanager
def managed_training_residency(model,*,cancel=None):
    from comfy import model_management as mm
    from comfy.model_patcher import ModelPatcher
    device=mm.get_torch_device()
    if device.type not in ('cpu','cuda'):
        raise ValueError('Training supports ComfyUI-selected CPU or CUDA')
    patcher=ModelPatcher(model,load_device=device,offload_device=mm.unet_offload_device())
    size=sum(t.numel()*t.element_size() for t in model.parameters())
    try:
        _check(cancel)
        mm.load_models_gpu([patcher],force_full_load=True,memory_required=size*4)
        _check(cancel)
        yield model
    finally:
        # Plain RopeND lazy tables are not model buffers and will not offload.
        for module in model.modules():
            for name in ('rope','rope_j','rope_t'):
                rope=getattr(module,name,None)
                for axis in range(getattr(rope,'nd',0)):
                    for prefix in ('cos_','sin_'):
                        key=f'{prefix}{axis}'
                        if hasattr(rope,key):
                            delattr(rope,key)
        mm.unload_model_and_clones(patcher,unload_additional_models=False)
