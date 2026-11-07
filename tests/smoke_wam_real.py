"""Opt-in one-candidate smoke with existing local assets; never controls hardware."""
import argparse
import json
from pathlib import Path

import numpy as np

from safe_wam.contracts import Frame, Observation
from safe_wam.efficient_wam import EfficientWAMPolicy, _json, _path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--safety-head',type=Path)
    args=parser.parse_args();cfg=_json(args.assets)
    stats=_json(_path(args.assets.resolve().parent,cfg['normalization']))
    state=np.asarray(stats['state']['mean'],dtype=np.float32)
    images={name:np.zeros(shape,dtype=np.uint8) for name,shape in cfg['camera_shapes'].items()}
    observation=Observation(tuple(Frame(images,t) for t in (-.2,-.133333333333,-.066666666667,0.)),state,0.)
    policy=EfficientWAMPolicy(args.assets,device=args.device)
    actions=policy.predict(observation)  # Exactly one WAM candidate-generation call.
    result=dict(shape=list(actions.shape),dtype=str(actions.dtype),finite=bool(np.isfinite(actions).all()),
                gripper_in_range=bool(((actions[:,6]>=0)&(actions[:,6]<=1)).all()),
                input='synthetic zero native RGB and original training mean state',
                generated_candidates=1,robot_commands_sent=0,checkpoint_step=cfg['checkpoint_step'])
    if args.safety_head:
        from safe_wam.pipeline import GuardedPolicy
        from safe_wam.safety_head import SafetyHead
        decision=GuardedPolicy(SafetyHead(args.safety_head,device='cpu')).evaluate(observation,actions=actions)
        result['pipeline_status']=decision.status;result['pipeline_reason']=decision.reason
    print(json.dumps(result))


if __name__=='__main__':main()
