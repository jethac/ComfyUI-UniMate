"""Public assembly publishes a complete numeric model for managed reload."""
import asyncio
from unittest.mock import patch

import pytest

from test_nodes import nodes,extension_module
import folder_paths
from test_trained_bundle import inputs
from unimate_pack.inference import load_model_bundle
from unimate_pack.bundle import inspect_bundle


def test_registered_assembly_schema():
    extension=asyncio.run(extension_module.comfy_entrypoint())
    classes=asyncio.run(extension.get_node_list())
    ids=[cls.GET_SCHEMA().node_id for cls in classes]
    assert len(ids)==len(set(ids))==37
    assert 'UniMateAssembleModel' in ids
    info=nodes.UniMateAssembleModel.GET_NODE_INFO_V1()
    assert info['output']==['UNIMATE_MODEL']
    assert info['output_node']
    assert 'workspace_mib' in nodes.UniMateModelLoader.GET_NODE_INFO_V1()['input']['optional']


def test_actual_managed_bundle_file_and_model_loader(tmp_path):
    output=tmp_path/'output'
    output.mkdir()
    values=inputs()
    with patch.object(folder_paths,'get_output_directory',return_value=str(output)):
        result=nodes.UniMateAssembleModel.execute(*values,sampling='{"method":"euler","num_steps":7}',
            filename_prefix='trained/model')
        model=result.result[0]
        descriptor=result.ui['files'][0]
        path=output/descriptor['subfolder']/descriptor['filename']
        reloaded=load_model_bundle(path)
        assert reloaded['bundle']==model['bundle'] and reloaded['sha256']==model['sha256']
        assert inspect_bundle(reloaded['bundle'])['solver']['method']=='euler'
        before=list(output.rglob('*.unimate'))
        with pytest.raises(ValueError):
            nodes.UniMateAssembleModel.execute(*values,filename_prefix='../escape')
        assert list(output.rglob('*.unimate'))==before
