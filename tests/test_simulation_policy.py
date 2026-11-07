from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from safe_wam.simulation_policy import SimulationEfficientWAMPolicy, native_observation


def native():
    return {'observation':{k:{'rgb':np.zeros((240,320,3),np.uint8)} for k in ('head_camera','left_camera','right_camera')},
            'joint_action':{'vector':np.arange(14,dtype=np.float32)}}


def test_constructor_is_lazy():
    policy=SimulationEfficientWAMPolicy('not_present.json')
    assert policy._runner is None
    subprocess.run([sys.executable,'-c',"import sys;from safe_wam.simulation_policy import SimulationEfficientWAMPolicy;SimulationEfficientWAMPolicy('absent');assert 'torch' not in sys.modules"],check=True)


def test_native_drive_proxy_not_replaced_or_normalized():
    observation=native()
    result=native_observation(observation)
    assert result is observation
    np.testing.assert_array_equal(result['joint_action']['vector'],np.arange(14,dtype=np.float32))
    public=SimpleNamespace(images={k:observation['observation'][n]['rgb'] for k,n in zip(('head','left','right'),('head_camera','left_camera','right_camera'))},
                           current_target14=observation['joint_action']['vector'],image_timestamp_s=0.,state_timestamp_s=0.)
    converted=native_observation(public)
    assert converted['joint_action']['vector'] is public.current_target14
    assert converted['observation']['left_camera']['rgb'] is public.images['left']


def test_seven_dimensional_state_and_invalid_camera_rejected_before_loading():
    policy=SimulationEfficientWAMPolicy('absent')
    observation=native();observation['joint_action']['vector']=np.zeros(7)
    with pytest.raises(ValueError):policy.predict(observation)
    assert policy._runner is None
    observation=native();observation['observation']['head_camera']['rgb']=np.zeros((128,128,3),np.uint8)
    with pytest.raises(ValueError):native_observation(observation)


def test_raw_absolute_targets_kept_and_wrong_family_rejected():
    class Stub(SimulationEfficientWAMPolicy):
        def _run(self,obs):return self.output
    policy=Stub('unused');policy._runner=object();policy._instruction='fixed cached task'
    policy.output=np.arange(224,dtype=np.float32).reshape(16,14)/100
    policy.output[0,6]=1.002;policy.output[0,13]=-.003
    result=policy.predict(native(),instruction='fixed cached task')
    np.testing.assert_array_equal(result,policy.output)
    assert policy._chunk_index==1
    with pytest.raises(ValueError):policy.predict(native(),instruction='different task')
    policy.output=np.zeros((16,7),np.float32)
    with pytest.raises(ValueError):policy.predict(native())


def test_episode_reset_clears_old_action_buffer():
    class Runner:
        def __init__(self):self.action_buffer=[1,2];self.observation_history=[3]
        def reset(self):self.action_buffer.clear();self.observation_history.clear()
        def set_instruction(self,text):self.text=text
    policy=SimulationEfficientWAMPolicy('unused');policy._runner=Runner();policy._instruction='cached'
    policy._chunk_index=9;policy.reset(seed=42)
    assert policy._chunk_index==0 and policy._seed==42 and policy._seed_override==42
    assert policy._runner.action_buffer==[] and policy._runner.observation_history==[] and policy._runner.text=='cached'
