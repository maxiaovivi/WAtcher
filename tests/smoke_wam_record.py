"""Opt-in single end-to-end candidate from four native frames in one LeRobot-v3 episode."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import av
    import pyarrow.parquet as pq
    import torch
    from safe_wam.contracts import Frame, Observation
    from safe_wam.efficient_wam import EfficientWAMPolicy
    from safe_wam.pipeline import GuardedPolicy
    from safe_wam.safety_head import SafetyHead
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--safety-head',type=Path,required=True)
    parser.add_argument('--episode',type=int,default=0);parser.add_argument('--first-frame',type=int,default=100)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();assert not a.output.exists()
    info=json.loads((a.dataset/'meta/info.json').read_text())
    # The smoke deliberately reads only the first metadata shard and one requested episode.
    episodes=pq.read_table(a.dataset/'meta/episodes/chunk-000/file-000.parquet',filters=[('episode_index','=',a.episode)])
    episode,=episodes.to_pylist();indices=list(range(a.first_frame,a.first_frame+4))
    path=a.dataset/info['data_path'].format(chunk_index=episode['data/chunk_index'],file_index=episode['data/file_index'])
    rows=pq.read_table(path,columns=['episode_index','frame_index','timestamp','observation.state','telemetry.source_timestamp_s','telemetry.camera_skew_ms'],
                       filters=[('episode_index','=',a.episode),('frame_index','in',indices)]).to_pylist()
    rows=sorted(rows,key=lambda r:r['frame_index']);assert [r['frame_index'] for r in rows]==indices
    actual_times=np.asarray([float(np.asarray(r['telemetry.source_timestamp_s']).reshape(-1)[0]) for r in rows])
    assert np.isfinite(actual_times).all() and (np.diff(actual_times)>0).all()
    images=[{} for _ in rows];video_proof={}
    for camera in ('external','hand','hand2'):
        key='observation.images.'+camera;prefix='videos/'+key
        offset=episode[prefix+'/from_timestamp']
        video=a.dataset/info['video_path'].format(video_key=key,chunk_index=episode[prefix+'/chunk_index'],file_index=episode[prefix+'/file_index'])
        targets=np.asarray([offset+r['timestamp'] for r in rows]);found={};pts=[]
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            container.seek(max(0,int((targets[0]-.1)/float(stream.time_base))),stream=stream,backward=True)
            for frame in container.decode(stream):
                timestamp=float(frame.pts*stream.time_base)
                for i,target in enumerate(targets):
                    if abs(timestamp-target)<1e-4:
                        assert i not in found;found[i]=frame.to_ndarray(format='rgb24');pts.append(timestamp)
                if timestamp>=targets[-1]-1e-4:break
        assert set(found)==set(range(4))
        for i in range(4):images[i][camera]=found[i]
        video_proof[camera]=dict(episode_video_start=offset,requested_pts=targets.tolist(),decoded_pts=pts,
                                 native_shape=list(found[0].shape))
    relative=actual_times-actual_times[-1]
    observation=Observation(tuple(Frame(rgb,float(ts)) for rgb,ts in zip(images,relative)),
                            np.asarray(rows[-1]['observation.state'],dtype=np.float32),0.)
    class RecordedPolicy:
        def __init__(self):self.inner=EfficientWAMPolicy(a.assets,device=a.device);self.actions=None;self.calls=0
        def predict(self,obs):
            self.calls+=1;self.actions=self.inner.predict(obs);return self.actions
    torch.manual_seed(17)
    policy=RecordedPolicy()
    decision=GuardedPolicy(SafetyHead(a.safety_head,device='cpu'),policy).evaluate(observation)
    result=dict(status='END_TO_END_REAL_RECORD_SMOKE',episode=a.episode,frame_indices=indices,
                relative_measured_times=relative.tolist(),camera_skew_ms=[r['telemetry.camera_skew_ms'] for r in rows],
                video_mapping=video_proof,seed=17,wam_predict_calls=policy.calls,decision=decision.status,reason=decision.reason,
                robot_commands_sent=0,source='native stored LeRobot video RGB and measured telemetry; no upsampled safety example',
                risk=None if decision.risk is None else dict(step=decision.risk.step.tolist(),post=decision.risk.post,overall=decision.risk.overall))
    if policy.actions is not None:
        actions=policy.actions
        result.update(candidate_shape=list(actions.shape),candidate_finite=bool(np.isfinite(actions).all()),
                      gripper_min=float(actions[:,6].min()),gripper_max=float(actions[:,6].max()),
                      candidate_actions=actions.tolist())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
