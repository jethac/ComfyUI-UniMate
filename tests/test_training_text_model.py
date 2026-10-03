"""Installed offline encoder outputs against pinned training-cache math."""
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import pytest

from unimate_pack.training_text import build_text_cache,text_views


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('UNIMATE_TEST_BUNDLE','UNIMATE_TEST_COMFY','UNIMATE_DATASET_REFERENCE')),
    reason='Requires explicit installed bundle, ComfyUI and pinned source')
def test_installed_training_cache_math_offline(monkeypatch):
    sys.path.insert(0,os.environ['UNIMATE_TEST_COMFY'])
    import torch
    from comfy import model_management as mm
    from unimate_pack.inference import load_model_bundle,_Runtime
    root=Path(os.environ['UNIMATE_DATASET_REFERENCE'])
    path=root/'unimate/utils/text_emb_cache.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='f1433ec15449a77721d921ebc16ae76ef8a552eed79b6413839370e64725dd01'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    names={'sequences_from_hidden','pooled_from_hidden','pool'}
    namespace={'np':np,'List':list}
    exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[]),str(path),'exec'),namespace)
    def forbidden(*args,**kwargs):
        raise AssertionError('Attempted network access')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(socket,'create_connection',forbidden)
    model=load_model_bundle(os.environ['UNIMATE_TEST_BUNDLE'])
    runtime=_Runtime(model)
    try:
        texts=['root','left knee','A character walks forward.','']
        inventory={key:value for key,value in runtime.manifest['files'].items() if key.startswith('text_encoder/')}
        identity=dict(encoder_type='t5',encoder_version=runtime.manifest['text_encoder']['id'],
            artifact_sha256=hashlib.sha256(json.dumps(dict(files=inventory,encoder=runtime.manifest['text_encoder']),sort_keys=True,separators=(',',':')).encode()).hexdigest())
        cache=build_text_cache(texts,runtime.encode,identity)
        inputs=runtime.tokenizer(texts,return_tensors='pt',padding=True)
        inputs['attention_mask'][-1]=0
        mm.load_models_gpu([runtime.encoder],force_full_load=True)
        inputs=inputs.to(runtime.encoder.load_device)
        with torch.inference_mode():
            hidden=runtime.encoder.model(**inputs).last_hidden_state
        expected_tokens=namespace['sequences_from_hidden'](hidden.cpu().numpy(),inputs['attention_mask'].cpu().numpy())
        expected_pooled=namespace['pooled_from_hidden'](hidden,inputs['attention_mask'])
        for index,text in enumerate(texts):
            tokens,pooled=text_views(cache,text,policy='cached')
            np.testing.assert_array_equal(tokens,expected_tokens[index])
            np.testing.assert_array_equal(pooled,expected_pooled[index])
            _,fresh=text_views(cache,text,policy='fresh')
            np.testing.assert_array_equal(fresh,namespace['pool'](expected_tokens[index]))
        rebuilt=build_text_cache(texts,forbidden,identity,existing=cache)
        assert rebuilt['arrays']==cache['arrays']
        print(json.dumps(dict(device=str(runtime.encoder.load_device),torch=torch.__version__,
            numpy=np.__version__,bundle=model['sha256'],encoder=identity,text_count=len(texts),
            cache_sha256=cache['sha256']),sort_keys=True))
    finally:
        mm.unload_all_models()
        runtime.directory.cleanup()
