"""Public training nodes using actual ComfyUI V3 schemas."""
import asyncio
import json

from test_nodes import nodes, extension_module
from test_training_dataset_samples import inputs
from test_training_text import IDENTITY,encoder
from unimate_pack.training_text import build_text_cache


def test_registered_training_schemas():
    extension=asyncio.run(extension_module.comfy_entrypoint())
    ids=[cls.GET_SCHEMA().node_id for cls in asyncio.run(extension.get_node_list())]
    assert len(ids)==len(set(ids))==29
    assert 'UniMateBuildTextCache' in ids
    assert 'UniMatePrepareTrainingSample' in ids
    assert nodes.UniMateBuildTextCache.GET_NODE_INFO_V1()['output']==['UNIMATE_TEXT_CACHE','STRING']
    assert nodes.UniMatePrepareTrainingSample.GET_NODE_INFO_V1()['output']==['UNIMATE_TRAINING_SAMPLE','STRING']


def test_actual_training_sample_node():
    from unimate_pack.training_sample_contracts import validate_training_sample
    dataset,stats,cache=inputs()
    result=nodes.UniMatePrepareTrainingSample.execute(dataset,stats,cache,'a',
        max_motion_length=8,max_joints=16,augmentation='addition',augmentation_seed=17)
    value=result.result[0]
    numeric=validate_training_sample(value)
    assert numeric['motion'].shape==(8,8,12)
    report=json.loads(result.result[1])
    assert report['provenance']['clip_id']=='a'


def test_text_node_delegates_model_and_dataset():
    from test_nodes import NodeTests
    tester=NodeTests()
    cache=build_text_cache(['root'],encoder,IDENTITY)
    captured={}
    def producer(model,dataset,**options):
        captured.update(model=model,dataset=dataset,options=options)
        return cache
    with tester.fake_module('training_text_model',build_dataset_text_cache=producer):
        result=nodes.UniMateBuildTextCache.execute({'model':True},{'dataset':True})
    assert result.result[0]==cache
    assert captured['model']=={'model':True}
    assert captured['options']['cancel'] is not None


def test_actual_codecs_preserve_training_values(tmp_path):
    from pathlib import Path
    from test_dataset_transport import codec
    from unimate_pack.training_sample_contracts import validate_training_sample
    from unimate_pack.training_text import validate_text_cache
    dataset,stats,cache=inputs()
    sample=nodes.UniMatePrepareTrainingSample.execute(dataset,stats,cache,'a',max_motion_length=8,max_joints=16).result[0]
    root=Path(__file__).resolve().parents[2]
    client=codec(root/'ComfyUI-Cloud-Offload/partition_protocol.py','training_client_codec')
    runner=codec(root/'cloud-offload/cloud_offload/partition_protocol.py','training_runner_codec')
    for kind,value,validate in [('UNIMATE_TEXT_CACHE',cache,validate_text_cache),
                              ('UNIMATE_TRAINING_SAMPLE',sample,validate_training_sample)]:
        client.validate_boundary_type(kind)
        runner.validate_boundary_type(kind)
        path=tmp_path/(kind+' client.partition')
        client.dump_bundle(client.pack_execution_values([value]),path)
        restored=runner.unpack_execution_values(runner.load_bundle(path))
        assert restored==[value]
        validate(restored[0])
        path=tmp_path/(kind+' worker.partition')
        runner.dump_bundle(runner.pack_execution_values(restored),path)
        assert client.unpack_execution_values(client.load_bundle(path))==[value]
