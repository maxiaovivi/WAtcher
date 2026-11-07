import subprocess
import sys

import numpy as np
import pytest

from safe_wam.efficient_wam import (
    EfficientWAMPolicy, _stats, _validate_manifest, denormalize_actions, normalize_state, prepare_mosaic,
)


def statistics():
    return {'state': {'min': [0.,0.,0.,0.,0.,0.,1.], 'max': [2.,2.,2.,2.,2.,2.,1.]},
            'action': {'min': [-2.]*6+[0.], 'max': [2.]*6+[1.]}}


def test_policy_construction_is_lazy(tmp_path):
    policy = EfficientWAMPolicy(tmp_path/'not_downloaded.json')
    assert policy._runner is None
    code = "import sys;from safe_wam.efficient_wam import EfficientWAMPolicy;EfficientWAMPolicy('absent');assert 'torch' not in sys.modules;assert 'av' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True)


def test_physical_normalization_constants_and_no_clipping():
    stats = _stats(statistics())
    state = np.array([[0.,1.,2.,3.,-1.,1.,1.]])
    np.testing.assert_array_equal(normalize_state(state,stats),[[-1.,0.,1.,2.,-2.,0.,0.]])
    predicted = np.array([[-1.,0.,1.,2.,-2.,0.,0.]],dtype=np.float32)
    np.testing.assert_array_equal(denormalize_actions(predicted,stats),[[-2.,0.,2.,4.,-4.,0.,.5]])


def test_invalid_stats_rejected():
    stats=statistics();stats['action']['min'][0]=float('nan')
    with pytest.raises(ValueError):_stats(stats)


def test_other_checkpoint_families_are_rejected():
    cfg={'prompt':'pick','task':'pick','checkpoint_step':14000,'statistics':statistics()}
    manifest=dict(variant='A',video_base_frozen=True,video_lora_parameters=0,video_lora_rank=None,
                  prompt='pick',task='pick',fps=15,action_horizon=16,cameras=['external','hand','hand2'],
                  normalization=statistics())
    _validate_manifest({'manifest':manifest,'step':14000},cfg)
    for key,value in [('variant','B'),('task','push'),('action_horizon',14)]:
        changed=dict(manifest,**{key:value})
        with pytest.raises(ValueError):_validate_manifest({'manifest':changed,'step':14000},cfg)
    with pytest.raises(ValueError):_validate_manifest({'manifest':manifest,'step':20000},cfg)
    with pytest.raises(ValueError):_validate_manifest({'format':'efficient_wam_push_goal_deploy_v1','manifest':manifest,'step':14000},cfg)


def test_three_camera_native_geometry_and_rgb_order():
    pytest.importorskip('av');pytest.importorskip('cv2')
    shapes={'external':[720,1280,3],'hand':[480,640,3],'hand2':[480,640,3]}
    colors={'external':[255,0,0],'hand':[0,255,0],'hand2':[0,0,255]}
    images={name:np.full(shape,colors[name],dtype=np.uint8) for name,shape in shapes.items()}
    output=prepare_mosaic(images,shapes)
    assert output.shape==(384,320,3) and output.dtype==np.uint8
    np.testing.assert_array_equal(output[150,160],[255,0,0])
    np.testing.assert_array_equal(output[300,80],[0,255,0])
    np.testing.assert_array_equal(output[300,240],[0,0,255])
    bad=dict(images,hand=images['hand'][:200])
    with pytest.raises(ValueError):prepare_mosaic(bad,shapes)


def test_invalid_observation_rejected_before_assets_are_loaded():
    from types import SimpleNamespace
    policy=EfficientWAMPolicy('absent')
    with pytest.raises(ValueError):policy.predict(SimpleNamespace(state=np.zeros(14),frames=[]))
    assert policy._runner is None
