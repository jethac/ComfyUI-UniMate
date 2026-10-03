"""Public workflow graph connects optimization, persistence and resume."""
from tools.training_execution_workflow import workflow_graph


def test_build_restore_reload_checkpoint_paths():
    for mode in ('build','restore','reload'):
        graph,inputs,outputs=workflow_graph(mode)
        assert graph['train']['class_type']=='UniMateTrain'
        assert graph['save']['class_type']=='UniMateSaveTrainingCheckpoint'
        assert graph['save']['inputs']['checkpoint']==['train',0]
        assert {x['key'] for x in outputs}=={'checkpoint','progress'}
        assert {'dataset','statistics','cache'}<={x['key'] for x in inputs}
        if mode=='build':
            assert 'checkpoint' not in graph['train']['inputs']
        elif mode=='restore':
            assert graph['train']['inputs']['checkpoint']==['in_checkpoint',0]
        else:
            assert graph['load']['class_type']=='UniMateLoadTrainingCheckpoint'
            assert graph['train']['inputs']['checkpoint']==['load',0]
