"""Bounded checkpoint files preserve numeric portable state without pickle."""
from copy import deepcopy
import struct

import pytest
import torch
from safetensors.torch import save

from test_training_checkpoint import checkpoint,rehash
from test_training_session import make
from unimate_pack import training_checkpoint_io as files


def value():
    session=make()
    return checkpoint(session)


def test_roundtrip_and_deterministic_file_bytes():
    original=value()
    payload=files.dump_training_checkpoint(original)
    assert payload.startswith(b'UMTRAIN1')
    assert files.load_training_checkpoint(payload)==original
    assert files.dump_training_checkpoint(files.load_training_checkpoint(payload))==payload


@pytest.mark.parametrize('kind',['magic','truncated','metadata_length','trailing','digest','tree','nonfinite'])
def test_bad_files_and_rehashed_state_rejected(kind):
    original=value()
    if kind=='digest':
        original['sha256']='f'*64
    elif kind=='tree':
        original['state']={'kind':'tensor','name':'t999999'}
        rehash(original)
    elif kind=='nonfinite':
        from safetensors.torch import load
        tensors=load(original['tensors'])
        target=next(t for t in tensors.values() if t.dtype==torch.float32 and t.numel())
        target.flatten()[0]=float('nan')
        original['tensors']=save(tensors)
        rehash(original)
    if kind in ('digest','tree','nonfinite'):
        with pytest.raises(ValueError):
            files.dump_training_checkpoint(original)
        return
    payload=files.dump_training_checkpoint(original)
    if kind=='magic':
        payload=b'BADMAGIC'+payload[8:]
    elif kind=='truncated':
        payload=payload[:10]
    elif kind=='metadata_length':
        payload=payload[:8]+struct.pack('<Q',2**63)+payload[16:]
    else:
        payload+=b'extra'
    with pytest.raises(ValueError):
        files.load_training_checkpoint(payload)


def test_workspace_and_cancellation_precede_decode(monkeypatch):
    original=value()
    payload=files.dump_training_checkpoint(original)
    def forbidden(*args,**kwargs):
        raise AssertionError('JSON decode before workspace check')
    monkeypatch.setattr(files,'_json',forbidden)
    with pytest.raises(ValueError,match='workspace'):
        files.load_training_checkpoint(payload,max_workspace_bytes=1)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        files.load_training_checkpoint(payload,cancel=cancel)


def test_file_validation_does_not_allocate_torch_state(monkeypatch):
    import safetensors.torch
    original=value()
    def forbidden(*args,**kwargs):
        raise AssertionError('numeric state allocation')
    monkeypatch.setattr(safetensors.torch,'load',forbidden)
    assert files.load_training_checkpoint(files.dump_training_checkpoint(original))==original


def test_binding_position_corruption_rejected_at_file_boundary():
    original=deepcopy(value())
    original['position']['consumed_batches']=True
    rehash(original)
    with pytest.raises(ValueError):
        files.dump_training_checkpoint(original)
