"""Explicit clip-mode scoring of a saved WAM candidate; no WAM model construction."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    import av
    import pyarrow.parquet as pq
    from safe_wam.contracts import Frame, Observation
    from safe_wam.pipeline import GuardedPolicy
    from safe_wam.safety_head import SafetyHead
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--saved-result',type=Path,required=True);parser.add_argument('--safety-head',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    saved=json.loads(a.saved_result.read_text());a.episode=saved['episode'];a.first_frame=saved['frame_indices'][0]
    assert saved['wam_predict_calls']==1 and saved['candidate_shape']==[16,7] and saved['candidate_finite'] is True
    assert not a.output.exists()
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
    assert video_proof==saved['video_mapping']
    assert relative.tolist()==saved['relative_measured_times']
    raw=np.asarray(saved['candidate_actions'],dtype=np.float32);raw.flags.writeable=False
    decision=GuardedPolicy(SafetyHead(a.safety_head,device='cpu'),gripper_mode='clip').evaluate(observation,actions=raw)
    result=decision.to_dict()
    result.update(status_origin='SAVED_RAW_WAM_CANDIDATE_EXPLICIT_CLIP_RESCORE',
                  original_status=saved['decision'],new_wam_generations=0,same_native_record_mapping_verified=True,
                  same_measured_relative_times_verified=True,raw_gripper_min=float(raw[:,6].min()),
                  raw_gripper_max=float(raw[:,6].max()),raw_candidate_actions=raw.tolist())
    if decision.candidate_actions is not None:
        assert np.array_equal(decision.candidate_actions[:,:6],raw[:,:6])
        assert np.array_equal(decision.candidate_actions[:,6],np.clip(raw[:,6],0,1))
        result['speed_columns_bitwise_unchanged']=True
        result['clipped_gripper_count']=int(np.count_nonzero(decision.candidate_actions[:,6]!=raw[:,6]))
    if decision.risk is not None:
        values=np.r_[decision.risk.step,decision.risk.post,decision.risk.overall]
        result['all_18_risk_probabilities_finite']=bool(np.isfinite(values).all())
        result['actual_gate_consistent']=decision.status==('REJECT' if decision.risk.overall>=decision.risk.threshold else 'ALLOW')
        result['approved_actions_same_scored_candidate']=bool(np.array_equal(decision.approved_actions,decision.candidate_actions)) if decision.allowed else decision.approved_actions is None
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
