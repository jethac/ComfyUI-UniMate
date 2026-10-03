"""Public source-compatible training text and encoded sample nodes."""
import json

from comfy_api.latest import io

from .dataset_nodes import CATEGORY,Dataset,Statistics,_cancel,_one,_LoadArchive,_save_archive,_input_path

Model=io.Custom('UNIMATE_MODEL')
TextCache=io.Custom('UNIMATE_TEXT_CACHE')
TrainingSample=io.Custom('UNIMATE_TRAINING_SAMPLE')
TrainingBatch=io.Custom('UNIMATE_TRAINING_BATCH')
TrainingJob=io.Custom('UNIMATE_TRAINING_JOB')
TrainingCheckpoint=io.Custom('UNIMATE_TRAINING_CHECKPOINT')
InferenceWeights=io.Custom('UNIMATE_INFERENCE_WEIGHTS')


def _weight_workspace(value):
    if type(value) is not int or not 1<=value<=65536:
        raise ValueError('Invalid workspace budget')
    return value*1024*1024


class UniMateAssembleModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Assemble UniMate Model',category=CATEGORY,
            inputs=[InferenceWeights.Input('weights'),Statistics.Input('statistics'),TextCache.Input('text_cache'),
                Model.Input('encoder_model',optional=True),
                io.String.Input('sampling',default='{}',multiline=True),
                io.String.Input('filename_prefix',default='unimate/trained-model'),
                io.Int.Input('workspace_mib',default=32768,min=1,max=65536)],
            outputs=[Model.Output()],is_output_node=True)

    @classmethod
    def execute(cls,weights,statistics,text_cache,encoder_model=None,sampling='{}',
        filename_prefix='unimate/trained-model',workspace_mib=32768):
        from .unimate_pack.trained_bundle import assemble_trained_bundle
        from .unimate_pack.bundle import _json
        value=assemble_trained_bundle(weights,statistics,text_cache,encoder_model,_json(sampling),
            cancel=_cancel,max_workspace_bytes=_weight_workspace(workspace_mib))
        return _save_archive(value,filename_prefix,'.unimate',lambda item:item['bundle'])


class UniMateExportInferenceWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Export UniMate Inference Weights',category=CATEGORY,
            inputs=[TrainingCheckpoint.Input('checkpoint'),io.Combo.Input('weights',options=['raw','ema']),
                io.String.Input('filename_prefix',default='unimate/trained'),
                io.Int.Input('workspace_mib',default=32768,min=1,max=65536)],
            outputs=[InferenceWeights.Output()],is_output_node=True)

    @classmethod
    def execute(cls,checkpoint,weights='ema',filename_prefix='unimate/trained',workspace_mib=32768):
        from .unimate_pack.inference_weights import make_inference_weights,dump_inference_weights
        budget=_weight_workspace(workspace_mib)
        value=make_inference_weights(checkpoint,weights,cancel=_cancel,max_workspace_bytes=budget)
        return _save_archive(value,filename_prefix,'.unimateweights',
            lambda item:dump_inference_weights(item,cancel=_cancel,max_workspace_bytes=budget))


class UniMateLoadInferenceWeights(_LoadArchive):
    from .unimate_pack.inference_weights import MAX_ARCHIVE_BYTES as LIMIT
    SUFFIX,TYPE,DISPLAY_NAME='.unimateweights',InferenceWeights,'Load UniMate Inference Weights'

    @classmethod
    def define_schema(cls):
        schema=super().define_schema()
        schema.inputs.append(io.Int.Input('workspace_mib',default=32768,min=1,max=65536))
        return schema

    @classmethod
    def execute(cls,archive,workspace_mib=32768):
        from .unimate_pack.inference_weights import load_inference_weights,_budget
        budget=_weight_workspace(workspace_mib)
        _cancel()
        path=_input_path(archive,cls.SUFFIX,cls.LIMIT)
        _budget(path.stat().st_size,budget)
        with path.open('rb') as stream:
            payload=stream.read(cls.LIMIT+1)
        value=load_inference_weights(payload,cancel=_cancel,max_workspace_bytes=budget)
        return io.NodeOutput(value)

    @classmethod
    def fingerprint_inputs(cls,archive,workspace_mib=32768):
        _weight_workspace(workspace_mib)
        return super().fingerprint_inputs(archive)


class UniMateLoadTrainingCheckpoint(_LoadArchive):
    from .unimate_pack.training_checkpoint_io import MAX_ARCHIVE_BYTES as LIMIT
    SUFFIX,TYPE,DISPLAY_NAME='.unimatetrain',TrainingCheckpoint,'Load UniMate Training Checkpoint'

    @staticmethod
    def load(payload):
        from .unimate_pack.training_checkpoint_io import load_training_checkpoint
        return load_training_checkpoint(payload,cancel=_cancel)


class UniMateSaveTrainingCheckpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Save UniMate Training Checkpoint',category=CATEGORY,
            inputs=[TrainingCheckpoint.Input('checkpoint'),
                io.String.Input('filename_prefix',default='unimate/training')],
            outputs=[TrainingCheckpoint.Output()],is_output_node=True)

    @classmethod
    def execute(cls,checkpoint,filename_prefix='unimate/training'):
        from .unimate_pack.training_checkpoint_io import dump_training_checkpoint
        return _save_archive(checkpoint,filename_prefix,'.unimatetrain',
            lambda value:dump_training_checkpoint(value,cancel=_cancel))


class UniMateTrain(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Train UniMate',category=CATEGORY,
            inputs=[TrainingJob.Input('job'),Dataset.Input('dataset'),Statistics.Input('statistics'),
                TextCache.Input('text_cache'),io.Int.Input('updates',default=1,min=1,max=10000),
                io.Int.Input('workspace_mib',default=8192,min=1,max=65536),
                TrainingCheckpoint.Input('checkpoint',optional=True),Model.Input('initialization',optional=True)],
            outputs=[TrainingCheckpoint.Output(),io.String.Output(display_name='training progress')])

    @classmethod
    def execute(cls,job,dataset,statistics,text_cache,updates=1,workspace_mib=8192,checkpoint=None,initialization=None):
        from functools import partial
        from comfy.utils import ProgressBar
        from .unimate_pack.training_execution import run_training_job
        from .unimate_pack.training_residency import managed_training_residency
        if type(workspace_mib) is not int or not 1<=workspace_mib<=65536:
            raise ValueError('Invalid workspace budget')
        if type(updates) is not int or not 1<=updates<=10000:
            raise ValueError('Invalid training chunk size')
        bar=ProgressBar(updates)
        value,report=run_training_job(job,dataset,statistics,text_cache,updates=updates,
            checkpoint=checkpoint,initialization=initialization,residency=partial(managed_training_residency,cancel=_cancel),
            cancel=_cancel,progress=lambda _:bar.update(1),max_workspace_bytes=workspace_mib*1024*1024)
        return io.NodeOutput(value,json.dumps(report,ensure_ascii=False,allow_nan=False))


class UniMateTrainingJob(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Configure UniMate Training',category=CATEGORY,
            inputs=[Dataset.Input('dataset'),Statistics.Input('statistics'),TextCache.Input('text_cache'),
                io.String.Input('options',default='{}',multiline=True),
                io.Int.Input('workspace_mib',default=512,min=1,max=65536),Model.Input('initialization',optional=True)],
            outputs=[TrainingJob.Output(),io.String.Output(display_name='job configuration')])

    @classmethod
    def execute(cls,dataset,statistics,text_cache,options='{}',workspace_mib=512,initialization=None):
        from .unimate_pack.bundle import _json
        from .unimate_pack.contracts import MAX_JSON_BYTES
        from .unimate_pack.training_job import make_training_job
        if type(options) is not str or len(options)>MAX_JSON_BYTES or len(options.encode('utf-8'))>MAX_JSON_BYTES:
            raise ValueError('Training options require bounded JSON')
        if type(workspace_mib) is not int or not 1<=workspace_mib<=65536:
            raise ValueError('Invalid workspace budget')
        value=make_training_job(dataset,statistics,text_cache,_json(options.encode('utf-8')),
            initialization=initialization,cancel=_cancel,max_workspace_bytes=workspace_mib*1024*1024)
        return io.NodeOutput(value,json.dumps(value,ensure_ascii=False,allow_nan=False))


class UniMateCollateTrainingSamples(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Collate UniMate Training Samples',category=CATEGORY,
            inputs=[TrainingSample.Input('samples'),io.Int.Input('workspace_mib',default=512,min=1,max=65536)],
            outputs=[TrainingBatch.Output(),io.String.Output(display_name='batch provenance')],is_input_list=True)

    @classmethod
    def execute(cls,samples,workspace_mib):
        from .unimate_pack.training_batch_contracts import collate_training_samples
        budget=_one(workspace_mib)
        if type(budget) is not int or not 1<=budget<=65536:
            raise ValueError('Invalid workspace budget')
        value=collate_training_samples(samples,max_workspace_bytes=budget*1024*1024,cancel=_cancel)
        report={key:item for key,item in value.items() if key!='arrays'}
        return io.NodeOutput(value,json.dumps(report,ensure_ascii=False,allow_nan=False))


class UniMateBuildTextCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Build UniMate Text Cache',category=CATEGORY,
            inputs=[Model.Input('model'),Dataset.Input('dataset'),
                    io.Int.Input('chunk_size',default=256,min=1,max=256),
                    TextCache.Input('existing',optional=True)],
            outputs=[TextCache.Output(),io.String.Output(display_name='cache provenance')])

    @classmethod
    def execute(cls,model,dataset,chunk_size=256,existing=None):
        from .unimate_pack.training_text_model import build_dataset_text_cache
        value=build_dataset_text_cache(model,dataset,existing=existing,
                                      chunk_size=chunk_size,cancel=_cancel)
        report={key:item for key,item in value.items() if key not in ('arrays','texts')}
        report['text_count']=len(value['texts'])
        return io.NodeOutput(value,json.dumps(report,ensure_ascii=False,allow_nan=False))


class UniMatePrepareTrainingSample(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Prepare UniMate Training Sample',category=CATEGORY,
            inputs=[Dataset.Input('dataset'),Statistics.Input('statistics'),TextCache.Input('text_cache'),
                io.String.Input('clip_id',default='clip1'),
                io.Combo.Input('mode',options=['tpos','first_frame']),
                io.Int.Input('max_motion_length',default=60,min=1,max=4096),
                io.Int.Input('max_joints',default=70,min=2,max=4096),
                io.Combo.Input('augmentation',options=['none','addition','addition_linear','removal','pooling','perturbation','random']),
                io.Int.Input('augmentation_seed',default=0,min=0,max=2**32-1),
                io.Int.Input('crop_seed',default=0,min=0,max=2**64-1),
                io.Int.Input('start_idx',default=-1,min=-1,max=2**31-1),
                io.Boolean.Input('realign_feature',default=True),io.Boolean.Input('ground_rest',default=True),
                io.Combo.Input('embedding_policy',options=['cached','fresh']),
                io.Combo.Input('addition_policy',options=['released','neutral_fk']),
                io.Boolean.Input('enable_addition',default=False),io.Boolean.Input('enable_removal',default=False),
                io.Boolean.Input('enable_pooling',default=False),io.Boolean.Input('enable_perturbation',default=False),
                io.Int.Input('max_freqs',default=8,min=1,max=4096),
                io.Int.Input('workspace_mib',default=512,min=1,max=65536)],
            outputs=[TrainingSample.Output(),io.String.Output(display_name='sample provenance')])

    @classmethod
    def execute(cls,dataset,statistics,text_cache,clip_id,mode='tpos',max_motion_length=60,max_joints=70,
        augmentation='none',augmentation_seed=0,crop_seed=0,start_idx=-1,realign_feature=True,
        ground_rest=True,embedding_policy='cached',addition_policy='released',enable_addition=False,
        enable_removal=False,enable_pooling=False,enable_perturbation=False,max_freqs=8,workspace_mib=512):
        from .unimate_pack.training_dataset_samples import produce_training_sample
        if type(start_idx) is not int or start_idx < -1:
            raise ValueError('start_idx must be -1 or a nonnegative frame index')
        if type(workspace_mib) is not int or not 1<=workspace_mib<=65536:
            raise ValueError('Invalid workspace budget')
        enabled=[]
        for operation,value in (('addition',enable_addition),('removal',enable_removal),
                                ('pooling',enable_pooling),('perturbation',enable_perturbation)):
            if type(value) is not bool:
                raise ValueError('Augmentation flags must be boolean')
            if value:
                enabled.append(operation)
        value=produce_training_sample(dataset,statistics,text_cache,clip_id,mode=mode,
            max_motion_length=max_motion_length,max_joints=max_joints,augmentation=augmentation,
            augmentation_seed=augmentation_seed,crop_seed=crop_seed,start_idx=None if start_idx==-1 else start_idx,
            realign_feature=realign_feature,ground_rest=ground_rest,embedding_policy=embedding_policy,
            addition_policy=addition_policy,enabled=tuple(enabled),max_freqs=max_freqs,cancel=_cancel,
            max_workspace_bytes=workspace_mib*1024*1024)
        report={key:item for key,item in value.items() if key!='arrays'}
        return io.NodeOutput(value,json.dumps(report,ensure_ascii=False,allow_nan=False))
