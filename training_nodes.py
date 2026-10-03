"""Public source-compatible training text and encoded sample nodes."""
import json

from comfy_api.latest import io

from .dataset_nodes import CATEGORY,Dataset,Statistics,_cancel,_one

Model=io.Custom('UNIMATE_MODEL')
TextCache=io.Custom('UNIMATE_TEXT_CACHE')
TrainingSample=io.Custom('UNIMATE_TRAINING_SAMPLE')
TrainingBatch=io.Custom('UNIMATE_TRAINING_BATCH')
TrainingJob=io.Custom('UNIMATE_TRAINING_JOB')


class UniMateTrainingJob(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='Configure UniMate Training',category=CATEGORY,
            inputs=[Dataset.Input('dataset'),Statistics.Input('statistics'),TextCache.Input('text_cache'),
                io.String.Input('options',default='{}',multiline=True),
                io.Int.Input('workspace_mib',default=512,min=1,max=65536)],
            outputs=[TrainingJob.Output(),io.String.Output(display_name='job configuration')])

    @classmethod
    def execute(cls,dataset,statistics,text_cache,options='{}',workspace_mib=512):
        from .unimate_pack.bundle import _json
        from .unimate_pack.contracts import MAX_JSON_BYTES
        from .unimate_pack.training_job import make_training_job
        if type(options) is not str or len(options)>MAX_JSON_BYTES or len(options.encode('utf-8'))>MAX_JSON_BYTES:
            raise ValueError('Training options require bounded JSON')
        if type(workspace_mib) is not int or not 1<=workspace_mib<=65536:
            raise ValueError('Invalid workspace budget')
        value=make_training_job(dataset,statistics,text_cache,_json(options.encode('utf-8')),
            cancel=_cancel,max_workspace_bytes=workspace_mib*1024*1024)
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
