import argparse
import json
import os
import os.path as osp
import pickle
from collections import OrderedDict

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


DEFAULT_CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def obtain_sensor2lidar(nusc,
                        sensor_token,
                        l2e_t,
                        l2e_r_mat,
                        e2g_t,
                        e2g_r_mat,
                        sensor_type):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = make_relative_path(nusc.get_sample_data_path(sd_rec['token']))

    sensor_info = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp'],
    }

    l2e_r_s = sensor_info['sensor2ego_rotation']
    l2e_t_s = sensor_info['sensor2ego_translation']
    e2g_r_s = sensor_info['ego2global_rotation']
    e2g_t_s = sensor_info['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    rotation = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    translation = (np.array(l2e_t_s) @ e2g_r_s_mat.T + np.array(e2g_t_s)) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    translation -= np.array(e2g_t) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    translation -= np.array(l2e_t) @ np.linalg.inv(l2e_r_mat).T

    sensor_info['sensor2lidar_rotation'] = rotation.T
    sensor_info['sensor2lidar_translation'] = translation
    return sensor_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create phase-specific frame-level infos for alocc2d_mini')
    parser.add_argument(
        '--canonical-info',
        default='',
        help=('optional canonical info pkl. When provided, generate only the '
              'semantic phase0/phase2/phase4 streams guided by canonical '
              'frame_tokens instead of the legacy raw-slot phase streams'))
    parser.add_argument(
        '--src-info',
        default='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
        help='source keyframe info pkl used to determine split scenes')
    parser.add_argument(
        '--data-root',
        default='data/nuscenes',
        help='nuScenes data root')
    parser.add_argument(
        '--version',
        default='v1.0-trainval',
        help='nuScenes version')
    parser.add_argument(
        '--out-dir',
        default='outputs/alocc2d_mini_phase_pkls',
        help='directory to save generated phase pkls')
    parser.add_argument(
        '--camera-names',
        nargs='+',
        default=DEFAULT_CAMERA_NAMES,
        help='camera names used to build each phase frame')
    parser.add_argument(
        '--token-camera',
        default='CAM_FRONT',
        help='camera whose sample_data token becomes info["token"]')
    parser.add_argument(
        '--canonical-phase-labels',
        nargs='+',
        type=int,
        default=[0, 2, 4],
        help=('phase file labels for canonical non-overlapping streams; '
              'default keeps the historical names phase0/phase2/phase4'))
    parser.add_argument(
        '--phase-count',
        type=int,
        default=6,
        help='number of phase streams to generate')
    parser.add_argument(
        '--frames-per-keyframe',
        type=int,
        default=6,
        help='expected sensor-frame stride between adjacent keyframes')
    parser.add_argument(
        '--max-camera-spread-ms',
        type=float,
        default=150.0,
        help='warn when timestamp spread within one 6-camera phase frame exceeds this value')
    parser.add_argument(
        '--max-lidar-dt-ms',
        type=float,
        default=75.0,
        help='max allowed time gap between phase frame and selected lidar')
    parser.add_argument(
        '--max-scenes',
        type=int,
        default=None,
        help='optional limit on number of scenes to process')
    parser.add_argument(
        '--allow-nonmonotonic-lidar',
        action='store_true',
        help='do not force lidar indices within one phase to be non-decreasing')
    parser.add_argument(
        '--skip-invalid-scenes',
        action='store_true',
        help='skip invalid scenes instead of raising immediately')
    parser.add_argument(
        '--planned-export-phases',
        nargs='+',
        type=int,
        default=None,
        help=('phase indices you plan to export together; for padded samples in '
              'these phases, reuse the real source token when it does not collide '
              'with another planned export phase'))
    return parser.parse_args()


def make_relative_path(path_str):
    path_str = str(path_str)
    cwd = os.getcwd()
    prefix = cwd + os.sep
    if path_str.startswith(prefix):
        return path_str[len(prefix):]
    return path_str


def load_pickle(path):
    if hasattr(mmcv, 'load'):
        return mmcv.load(path, file_format='pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(data, path):
    if hasattr(mmcv, 'dump'):
        return mmcv.dump(data, path)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def mkdir_or_exist(path):
    if hasattr(mmcv, 'mkdir_or_exist'):
        mmcv.mkdir_or_exist(path)
        return
    os.makedirs(path, exist_ok=True)


def track_progress(items):
    if hasattr(mmcv, 'track_iter_progress'):
        return mmcv.track_iter_progress(items)
    return items


def load_base_info(src_info):
    data = load_pickle(src_info)
    infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    metadata = data.get('metadata', {})
    return infos, metadata


def load_canonical_info(canonical_info):
    data = load_pickle(canonical_info)
    infos = data['infos'] if isinstance(data, dict) else data
    metadata = data.get('metadata', {}) if isinstance(data, dict) else {}
    grouped = OrderedDict()
    for info in infos:
        scene_token = info['scene_token']
        if scene_token not in grouped:
            grouped[scene_token] = {
                'scene_name': info['scene_name'],
                'infos': [],
                'by_token': {},
            }
        grouped[scene_token]['infos'].append(info)
        grouped[scene_token]['by_token'][info['token']] = info
    return grouped, metadata


def build_synthetic_export_token(token, phase_idx, source_phase_slot,
                                 keyframe_token):
    return (
        f'{token}__pad_p{phase_idx}_from{source_phase_slot}_'
        f'kf_{keyframe_token}')


def ensure_scene_fields(nusc, info):
    if 'scene_token' in info and 'scene_name' in info:
        return info

    info = dict(info)
    sample = nusc.get('sample', info['token'])
    scene_token = sample['scene_token']
    scene = nusc.get('scene', scene_token)
    info['scene_token'] = scene_token
    info['scene_name'] = scene['name']
    return info


def group_keyframe_infos_by_scene(nusc, infos):
    grouped = OrderedDict()
    for info in infos:
        info = ensure_scene_fields(nusc, info)
        scene_token = info['scene_token']
        if scene_token not in grouped:
            grouped[scene_token] = {
                'scene_name': info['scene_name'],
                'infos': [],
            }
        grouped[scene_token]['infos'].append(info)
    return grouped


def sample_data_belongs_to_scene(nusc, sd_rec, scene_token):
    sample = nusc.get('sample', sd_rec['sample_token'])
    return sample['scene_token'] == scene_token


def build_sensor_chain(nusc, start_token, expected_channel, scene_token):
    chain = []
    token = start_token
    while token:
        sd_rec = nusc.get('sample_data', token)
        if sd_rec['channel'] != expected_channel:
            raise ValueError(
                f'Expected channel {expected_channel}, got {sd_rec["channel"]}')
        if not sample_data_belongs_to_scene(nusc, sd_rec, scene_token):
            break
        chain.append(sd_rec)
        token = sd_rec['next']
    return chain


def validate_camera_chains(camera_chains, scene_keyframe_infos,
                           frames_per_keyframe, camera_names):
    expected_keyframe_count = len(scene_keyframe_infos)
    expected_key_indices = [
        frames_per_keyframe * i for i in range(expected_keyframe_count)
    ]

    lengths = {cam_name: len(chain) for cam_name, chain in camera_chains.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f'Camera chain lengths mismatch: {lengths}')

    for cam_name in camera_names:
        chain = camera_chains[cam_name]
        key_indices = [i for i, sd in enumerate(chain) if sd['is_key_frame']]
        if key_indices != expected_key_indices:
            raise ValueError(
                f'{cam_name} keyframe indices mismatch. '
                f'Expected {expected_key_indices[:10]}..., got {key_indices[:10]}...')

        for idx, base_info in zip(expected_key_indices, scene_keyframe_infos):
            if chain[idx]['sample_token'] != base_info['token']:
                raise ValueError(
                    f'{cam_name} keyframe sample token mismatch at index {idx}: '
                    f'{chain[idx]["sample_token"]} != {base_info["token"]}')

        for idx, sd in enumerate(chain):
            should_be_keyframe = (idx % frames_per_keyframe == 0)
            if bool(sd['is_key_frame']) != should_be_keyframe:
                raise ValueError(
                    f'{cam_name} frame {idx} keyframe flag mismatch: '
                    f'expected {should_be_keyframe}, got {sd["is_key_frame"]}')


def build_lidar_chain(nusc, scene_token):
    scene = nusc.get('scene', scene_token)
    first_sample = nusc.get('sample', scene['first_sample_token'])
    start_token = first_sample['data']['LIDAR_TOP']
    return build_sensor_chain(
        nusc=nusc,
        start_token=start_token,
        expected_channel='LIDAR_TOP',
        scene_token=scene_token)


def build_camera_chains(nusc, scene_token, camera_names):
    scene = nusc.get('scene', scene_token)
    first_sample = nusc.get('sample', scene['first_sample_token'])
    chains = OrderedDict()
    for cam_name in camera_names:
        start_token = first_sample['data'][cam_name]
        chains[cam_name] = build_sensor_chain(
            nusc=nusc,
            start_token=start_token,
            expected_channel=cam_name,
            scene_token=scene_token)
    return chains


def camera_bundle_timestamp_stats(cam_bundle):
    timestamps = [cam_bundle[cam_name]['timestamp'] for cam_name in cam_bundle]
    timestamps = np.asarray(timestamps, dtype=np.int64)
    return {
        'min': int(timestamps.min()),
        'max': int(timestamps.max()),
        'median': int(np.median(timestamps)),
        'spread': int(timestamps.max() - timestamps.min()),
    }


def find_nearest_lidar_index(lidar_timestamps, anchor_timestamp, min_idx=None):
    if min_idx is None:
        min_idx = 0
    min_idx = max(0, min_idx)
    if min_idx >= len(lidar_timestamps):
        return len(lidar_timestamps) - 1

    search_space = np.asarray(lidar_timestamps[min_idx:], dtype=np.int64)
    rel_idx = int(np.argmin(np.abs(search_space - int(anchor_timestamp))))
    return min_idx + rel_idx


def build_lidar_reference_info(nusc, lidar_sd):
    cs_record = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    l2e_t = cs_record['translation']
    l2e_r = cs_record['rotation']
    e2g_t = pose_record['translation']
    e2g_r = pose_record['rotation']

    return {
        'lidar_token': lidar_sd['token'],
        'lidar_path': make_relative_path(nusc.get_sample_data_path(lidar_sd['token'])),
        'timestamp': int(lidar_sd['timestamp']),
        'lidar2ego_translation': l2e_t,
        'lidar2ego_rotation': l2e_r,
        'ego2global_translation': e2g_t,
        'ego2global_rotation': e2g_r,
        'l2e_t': np.array(l2e_t),
        'l2e_r_mat': Quaternion(l2e_r).rotation_matrix,
        'e2g_t': np.array(e2g_t),
        'e2g_r_mat': Quaternion(e2g_r).rotation_matrix,
    }


def build_camera_entry_against_ref_lidar(nusc, cam_sd_token, cam_name,
                                         ref_lidar_info):
    cam_info = obtain_sensor2lidar(
        nusc,
        cam_sd_token,
        ref_lidar_info['l2e_t'],
        ref_lidar_info['l2e_r_mat'],
        ref_lidar_info['e2g_t'],
        ref_lidar_info['e2g_r_mat'],
        cam_name)
    cam_intrinsic = nusc.get('calibrated_sensor', nusc.get(
        'sample_data', cam_sd_token)['calibrated_sensor_token'])['camera_intrinsic']
    cam_info.update(cam_intrinsic=np.array(cam_intrinsic))
    return cam_info


def build_placeholder_occ_path(scene_name, token):
    return osp.join('./data/nuscenes/gts', scene_name, token)


def collect_interval_sensor_frames(nusc, start_token, end_token, expected_channel,
                                   scene_token, max_frames):
    start_sd = nusc.get('sample_data', start_token)
    end_sd = nusc.get('sample_data', end_token)
    if not start_sd['is_key_frame']:
        raise ValueError(
            f'Interval start token {start_token} for {expected_channel} is not a keyframe')
    if not end_sd['is_key_frame']:
        raise ValueError(
            f'Interval end token {end_token} for {expected_channel} is not a keyframe')

    frames = []
    token = start_token
    while token and token != end_token:
        sd_rec = nusc.get('sample_data', token)
        if sd_rec['channel'] != expected_channel:
            raise ValueError(
                f'Expected channel {expected_channel}, got {sd_rec["channel"]}')
        if not sample_data_belongs_to_scene(nusc, sd_rec, scene_token):
            raise ValueError(
                f'{expected_channel} interval crossed scene boundary before reaching next keyframe')
        frames.append(sd_rec)
        token = sd_rec['next']

    if token != end_token:
        raise ValueError(
            f'Failed to reach next keyframe for {expected_channel}: {start_token} -> {end_token}')
    if max_frames is not None and len(frames) > max_frames:
        print(
            f'[WARN] {expected_channel} interval contains {len(frames)} frames before next keyframe; '
            f'only the first {max_frames} phase slots will be used')
    return frames


def find_nearest_sensor_frame(frames, anchor_timestamp):
    if not frames:
        raise ValueError('Cannot choose nearest frame from an empty interval')
    timestamps = np.asarray([frame['timestamp'] for frame in frames], dtype=np.int64)
    rel_idx = int(np.argmin(np.abs(timestamps - int(anchor_timestamp))))
    return frames[rel_idx], rel_idx


def build_phase_info(nusc, scene_name, scene_token, phase_idx,
                     frame_idx_in_phase, frame_global_index, cam_bundle,
                     ref_lidar_info, keyframe_token, next_keyframe_token,
                     is_keyframe, token_camera, source_phase_slot,
                     usable_phase_count, export_token,
                     canonical_phase_idx=None,
                     canonical_step_index=None,
                     interval_idx=None,
                     is_phase_padded=None):
    cams = OrderedDict()
    for cam_name, cam_sd in cam_bundle.items():
        cams[cam_name] = build_camera_entry_against_ref_lidar(
            nusc=nusc,
            cam_sd_token=cam_sd['token'],
            cam_name=cam_name,
            ref_lidar_info=ref_lidar_info)

    timestamp_stats = camera_bundle_timestamp_stats(cam_bundle)
    token = cams[token_camera]['sample_data_token']

    info = {
        'token': token,
        'export_token': export_token,
        'scene_name': scene_name,
        'scene_token': scene_token,
        'timestamp': timestamp_stats['median'],
        'anchor_timestamp': timestamp_stats['median'],
        'camera_timestamp_min': timestamp_stats['min'],
        'camera_timestamp_max': timestamp_stats['max'],
        'camera_timestamp_spread': timestamp_stats['spread'],
        'phase_idx': phase_idx,
        'frame_idx_in_phase': frame_idx_in_phase,
        'frame_global_index': frame_global_index,
        'keyframe_token': keyframe_token,
        'next_keyframe_token': next_keyframe_token,
        'is_keyframe': bool(is_keyframe),
        'source_phase_slot': int(source_phase_slot),
        'usable_phase_count': int(usable_phase_count),
        'is_phase_padded': (
            bool(source_phase_slot != phase_idx)
            if is_phase_padded is None else bool(is_phase_padded)
        ),
        'lidar_token': ref_lidar_info['lidar_token'],
        'lidar_path': ref_lidar_info['lidar_path'],
        'lidar2ego_translation': ref_lidar_info['lidar2ego_translation'],
        'lidar2ego_rotation': ref_lidar_info['lidar2ego_rotation'],
        'ego2global_translation': ref_lidar_info['ego2global_translation'],
        'ego2global_rotation': ref_lidar_info['ego2global_rotation'],
        'sweeps': [],
        'cams': cams,
        'occ_path': build_placeholder_occ_path(scene_name, keyframe_token),
    }
    if canonical_phase_idx is not None:
        info['canonical_phase_idx'] = int(canonical_phase_idx)
    if canonical_step_index is not None:
        info['canonical_step_index'] = int(canonical_step_index)
    if interval_idx is not None:
        info['interval_idx'] = int(interval_idx)
    return info


def validate_canonical_settings(canonical_metadata, canonical_example, phase_labels):
    if len(phase_labels) != 3:
        raise ValueError('--canonical-phase-labels must contain exactly 3 labels')
    if len(set(phase_labels)) != 3:
        raise ValueError('--canonical-phase-labels contains duplicates')

    history_keyframes = int(
        canonical_metadata.get('history_keyframes',
                               canonical_example.get('history_keyframes', 4)))
    steps_per_interval = int(
        canonical_metadata.get('steps_per_interval',
                               canonical_example.get('steps_per_interval', 3)))
    if history_keyframes != 4:
        raise ValueError(
            f'Canonical mode currently expects history_keyframes=4, got {history_keyframes}')
    if steps_per_interval != 3:
        raise ValueError(
            f'Canonical mode currently expects steps_per_interval=3, got {steps_per_interval}')
    return history_keyframes, steps_per_interval


def validate_scene_alignment(scene_name, scene_keyframe_infos, canonical_entries):
    if len(scene_keyframe_infos) != len(canonical_entries):
        raise ValueError(
            f'Scene {scene_name} keyframe count mismatch: '
            f'bevdetv2={len(scene_keyframe_infos)} canonical={len(canonical_entries)}')
    for base_info, canonical_info in zip(scene_keyframe_infos, canonical_entries):
        if base_info['token'] != canonical_info['token']:
            raise ValueError(
                f'Scene {scene_name} token mismatch: '
                f'bevdetv2={base_info["token"]} canonical={canonical_info["token"]}')


def precompute_interval_frames(nusc, scene_token, scene_keyframe_infos, camera_names):
    interval_frames = []
    for interval_idx in range(len(scene_keyframe_infos) - 1):
        curr_keyframe_token = scene_keyframe_infos[interval_idx]['token']
        next_keyframe_token = scene_keyframe_infos[interval_idx + 1]['token']
        curr_sample = nusc.get('sample', curr_keyframe_token)
        next_sample = nusc.get('sample', next_keyframe_token)
        frames_by_cam = OrderedDict()
        for cam_name in camera_names:
            frames_by_cam[cam_name] = collect_interval_sensor_frames(
                nusc=nusc,
                start_token=curr_sample['data'][cam_name],
                end_token=next_sample['data'][cam_name],
                expected_channel=cam_name,
                scene_token=scene_token,
                max_frames=None)
        interval_frames.append({
            'start_keyframe_token': curr_keyframe_token,
            'end_keyframe_token': next_keyframe_token,
            'frames_by_cam': frames_by_cam,
        })
    return interval_frames


def build_keyframe_phase_info(nusc, scene_name, scene_token, phase_label,
                              canonical_phase_idx, frame_idx_in_phase,
                              frame_global_index, keyframe_sample_token,
                              next_keyframe_token, token_camera,
                              camera_names):
    sample = nusc.get('sample', keyframe_sample_token)
    cam_bundle = OrderedDict(
        (cam_name, nusc.get('sample_data', sample['data'][cam_name]))
        for cam_name in camera_names)
    selected_token = cam_bundle[token_camera]['token']
    ref_lidar_info = build_lidar_reference_info(
        nusc, nusc.get('sample_data', sample['data']['LIDAR_TOP']))
    return build_phase_info(
        nusc=nusc,
        scene_name=scene_name,
        scene_token=scene_token,
        phase_idx=phase_label,
        frame_idx_in_phase=frame_idx_in_phase,
        frame_global_index=frame_global_index,
        cam_bundle=cam_bundle,
        ref_lidar_info=ref_lidar_info,
        keyframe_token=keyframe_sample_token,
        next_keyframe_token=next_keyframe_token,
        is_keyframe=True,
        token_camera=token_camera,
        source_phase_slot=0,
        usable_phase_count=1,
        export_token=selected_token,
        canonical_phase_idx=canonical_phase_idx,
        canonical_step_index=frame_global_index,
        interval_idx=frame_global_index // 3 if frame_global_index > 0 else 0,
        is_phase_padded=False)


def build_canonical_interval_phase_info(nusc, scene_name, scene_token, phase_label,
                                        canonical_phase_idx, frame_idx_in_phase,
                                        frame_global_index, interval_idx,
                                        interval_record, selected_token,
                                        selected_local_index, token_camera,
                                        args):
    frames_by_cam = interval_record['frames_by_cam']
    cam_front_frames = frames_by_cam[args.token_camera]
    if selected_local_index < 0 or selected_local_index >= len(cam_front_frames):
        raise ValueError(
            f'Invalid canonical local index {selected_local_index} in scene {scene_name}, '
            f'interval {interval_idx}')
    if cam_front_frames[selected_local_index]['token'] != selected_token:
        raise ValueError(
            f'Canonical token mismatch in scene {scene_name}, interval {interval_idx}: '
            f'expected {selected_token}, got {cam_front_frames[selected_local_index]["token"]}')
    anchor_sd = nusc.get('sample_data', selected_token)
    anchor_timestamp = int(anchor_sd['timestamp'])

    cam_bundle = OrderedDict()
    camera_local_indices = {}
    for cam_name in args.camera_names:
        if cam_name == args.token_camera:
            cam_bundle[cam_name] = anchor_sd
            camera_local_indices[cam_name] = int(selected_local_index)
            continue
        chosen_sd, local_idx = find_nearest_sensor_frame(
            frames=frames_by_cam[cam_name],
            anchor_timestamp=anchor_timestamp)
        cam_bundle[cam_name] = chosen_sd
        camera_local_indices[cam_name] = int(local_idx)

    ts_stats = camera_bundle_timestamp_stats(cam_bundle)
    if ts_stats['spread'] > int(args.max_camera_spread_ms * 1000):
        print(
            f'[WARN] Camera timestamp spread is large in scene {scene_name}, '
            f'phase {phase_label}, frame {frame_idx_in_phase}: {ts_stats["spread"]} us')

    min_lidar_idx = None
    if not args.allow_nonmonotonic_lidar:
        min_lidar_idx = args._last_lidar_indices[phase_label]
    lidar_idx = find_nearest_lidar_index(
        lidar_timestamps=args._lidar_timestamps,
        anchor_timestamp=anchor_timestamp,
        min_idx=min_lidar_idx)
    ref_lidar_info = build_lidar_reference_info(nusc, args._lidar_chain[lidar_idx])

    lidar_dt = abs(ref_lidar_info['timestamp'] - anchor_timestamp)
    if lidar_dt > int(args.max_lidar_dt_ms * 1000):
        raise ValueError(
            f'Lidar time gap too large in scene {scene_name}, '
            f'phase {phase_label}, frame {frame_idx_in_phase}: {lidar_dt} us')

    info = build_phase_info(
        nusc=nusc,
        scene_name=scene_name,
        scene_token=scene_token,
        phase_idx=phase_label,
        frame_idx_in_phase=frame_idx_in_phase,
        frame_global_index=frame_global_index,
        cam_bundle=cam_bundle,
        ref_lidar_info=ref_lidar_info,
        keyframe_token=interval_record['start_keyframe_token'],
        next_keyframe_token=interval_record['end_keyframe_token'],
        is_keyframe=False,
        token_camera=token_camera,
        source_phase_slot=selected_local_index,
        usable_phase_count=len(cam_front_frames),
        export_token=selected_token,
        canonical_phase_idx=canonical_phase_idx,
        canonical_step_index=frame_global_index,
        interval_idx=interval_idx,
        is_phase_padded=False)
    info['anchor_timestamp'] = anchor_timestamp
    info['camera_source_local_indices'] = camera_local_indices
    args._last_lidar_indices[phase_label] = lidar_idx
    return info


def process_one_scene_with_canonical(nusc, scene_token, scene_name,
                                     scene_keyframe_infos, scene_canonical_infos,
                                     args, history_keyframes,
                                     steps_per_interval, phase_labels):
    validate_scene_alignment(
        scene_name=scene_name,
        scene_keyframe_infos=scene_keyframe_infos,
        canonical_entries=scene_canonical_infos)

    interval_frames = precompute_interval_frames(
        nusc=nusc,
        scene_token=scene_token,
        scene_keyframe_infos=scene_keyframe_infos,
        camera_names=args.camera_names)

    lidar_chain = build_lidar_chain(nusc, scene_token)
    if not lidar_chain:
        raise ValueError(f'No lidar chain found for scene {scene_name}')

    args._lidar_chain = lidar_chain
    args._lidar_timestamps = [sd['timestamp'] for sd in lidar_chain]
    args._last_lidar_indices = {phase_label: 0 for phase_label in phase_labels}

    phase_infos = {phase_label: [] for phase_label in phase_labels}
    phase0_label, phase1_label, phase2_label = phase_labels

    nonempty_entries = [
        (local_idx, info)
        for local_idx, info in enumerate(scene_canonical_infos)
        if len(info.get('frame_tokens', [])) > 0
    ]
    if not nonempty_entries:
        del args._lidar_chain
        del args._lidar_timestamps
        del args._last_lidar_indices
        return phase_infos

    first_local_idx, first_entry = nonempty_entries[0]
    if first_local_idx < history_keyframes:
        raise ValueError(
            f'Scene {scene_name} first canonical entry with frame_tokens appears too early: '
            f'{first_local_idx}')

    keyframe_positions = [steps_per_interval * idx for idx in range(history_keyframes + 1)]
    phase1_positions = [steps_per_interval * idx + 1 for idx in range(history_keyframes)]
    phase2_positions = [steps_per_interval * idx + 2 for idx in range(history_keyframes)]
    keyframe_sample_tokens = list(first_entry['keyframe_sample_tokens'])
    first_frame_tokens = list(first_entry['frame_tokens'])
    first_local_indices = [
        [int(v) for v in row]
        for row in np.asarray(first_entry['interval_selected_local_indices']).tolist()
    ]

    for offset, frame_pos in enumerate(keyframe_positions):
        scene_keyframe_idx = first_local_idx - history_keyframes + offset
        next_keyframe_token = None
        if scene_keyframe_idx + 1 < len(scene_keyframe_infos):
            next_keyframe_token = scene_keyframe_infos[scene_keyframe_idx + 1]['token']
        phase_infos[phase0_label].append(
            build_keyframe_phase_info(
                nusc=nusc,
                scene_name=scene_name,
                scene_token=scene_token,
                phase_label=phase0_label,
                canonical_phase_idx=0,
                frame_idx_in_phase=len(phase_infos[phase0_label]),
                frame_global_index=scene_keyframe_idx * steps_per_interval,
                keyframe_sample_token=keyframe_sample_tokens[offset],
                next_keyframe_token=next_keyframe_token,
                token_camera=args.token_camera,
                camera_names=args.camera_names))

    for interval_offset, frame_pos in enumerate(phase1_positions):
        interval_idx = first_local_idx - history_keyframes + interval_offset
        selected_token = first_frame_tokens[frame_pos]
        selected_local_index = int(first_local_indices[interval_offset][1])
        phase_infos[phase1_label].append(
            build_canonical_interval_phase_info(
                nusc=nusc,
                scene_name=scene_name,
                scene_token=scene_token,
                phase_label=phase1_label,
                canonical_phase_idx=1,
                frame_idx_in_phase=len(phase_infos[phase1_label]),
                frame_global_index=interval_idx * steps_per_interval + 1,
                interval_idx=interval_idx,
                interval_record=interval_frames[interval_idx],
                selected_token=selected_token,
                selected_local_index=selected_local_index,
                token_camera=args.token_camera,
                args=args))

    for interval_offset, frame_pos in enumerate(phase2_positions):
        interval_idx = first_local_idx - history_keyframes + interval_offset
        selected_token = first_frame_tokens[frame_pos]
        selected_local_index = int(first_local_indices[interval_offset][2])
        phase_infos[phase2_label].append(
            build_canonical_interval_phase_info(
                nusc=nusc,
                scene_name=scene_name,
                scene_token=scene_token,
                phase_label=phase2_label,
                canonical_phase_idx=2,
                frame_idx_in_phase=len(phase_infos[phase2_label]),
                frame_global_index=interval_idx * steps_per_interval + 2,
                interval_idx=interval_idx,
                interval_record=interval_frames[interval_idx],
                selected_token=selected_token,
                selected_local_index=selected_local_index,
                token_camera=args.token_camera,
                args=args))

    for local_idx, entry in nonempty_entries[1:]:
        current_keyframe_token = scene_keyframe_infos[local_idx]['token']
        next_keyframe_token = None
        if local_idx + 1 < len(scene_keyframe_infos):
            next_keyframe_token = scene_keyframe_infos[local_idx + 1]['token']
        phase_infos[phase0_label].append(
            build_keyframe_phase_info(
                nusc=nusc,
                scene_name=scene_name,
                scene_token=scene_token,
                phase_label=phase0_label,
                canonical_phase_idx=0,
                frame_idx_in_phase=len(phase_infos[phase0_label]),
                frame_global_index=local_idx * steps_per_interval,
                keyframe_sample_token=current_keyframe_token,
                next_keyframe_token=next_keyframe_token,
                token_camera=args.token_camera,
                camera_names=args.camera_names))

        interval_idx = local_idx - 1
        entry_local_indices = [
            [int(v) for v in row]
            for row in np.asarray(entry['interval_selected_local_indices']).tolist()
        ]
        phase_infos[phase1_label].append(
            build_canonical_interval_phase_info(
                nusc=nusc,
                scene_name=scene_name,
                scene_token=scene_token,
                phase_label=phase1_label,
                canonical_phase_idx=1,
                frame_idx_in_phase=len(phase_infos[phase1_label]),
                frame_global_index=interval_idx * steps_per_interval + 1,
                interval_idx=interval_idx,
                interval_record=interval_frames[interval_idx],
                selected_token=entry['frame_tokens'][-3],
                selected_local_index=int(entry_local_indices[-1][1]),
                token_camera=args.token_camera,
                args=args))
        phase_infos[phase2_label].append(
            build_canonical_interval_phase_info(
                nusc=nusc,
                scene_name=scene_name,
                scene_token=scene_token,
                phase_label=phase2_label,
                canonical_phase_idx=2,
                frame_idx_in_phase=len(phase_infos[phase2_label]),
                frame_global_index=interval_idx * steps_per_interval + 2,
                interval_idx=interval_idx,
                interval_record=interval_frames[interval_idx],
                selected_token=entry['frame_tokens'][-2],
                selected_local_index=int(entry_local_indices[-1][2]),
                token_camera=args.token_camera,
                args=args))

    del args._lidar_chain
    del args._lidar_timestamps
    del args._last_lidar_indices
    return phase_infos


def process_one_scene(nusc, scene_token, scene_name, scene_keyframe_infos, args):
    lidar_chain = build_lidar_chain(nusc, scene_token)
    lidar_timestamps = [sd['timestamp'] for sd in lidar_chain]

    if not lidar_chain:
        raise ValueError(f'No lidar chain found for scene {scene_name}')

    phase_infos = {phase_idx: [] for phase_idx in range(args.phase_count)}
    last_lidar_indices = {phase_idx: 0 for phase_idx in range(args.phase_count)}

    for interval_idx in range(len(scene_keyframe_infos) - 1):
        curr_keyframe_token = scene_keyframe_infos[interval_idx]['token']
        next_keyframe_token = scene_keyframe_infos[interval_idx + 1]['token']
        curr_sample = nusc.get('sample', curr_keyframe_token)
        next_sample = nusc.get('sample', next_keyframe_token)

        interval_frames_by_cam = OrderedDict()
        for cam_name in args.camera_names:
            interval_frames_by_cam[cam_name] = collect_interval_sensor_frames(
                nusc=nusc,
                start_token=curr_sample['data'][cam_name],
                end_token=next_sample['data'][cam_name],
                expected_channel=cam_name,
                scene_token=scene_token,
                max_frames=args.frames_per_keyframe)

        usable_phase_count = min(
            min(len(interval_frames_by_cam[cam_name]), args.frames_per_keyframe)
            for cam_name in args.camera_names)
        if usable_phase_count <= 0:
            raise ValueError(
                f'No usable phase slots found in scene {scene_name}, interval {interval_idx}')
        if usable_phase_count < args.phase_count:
            print(
                f'[WARN] scene {scene_name}, interval {interval_idx} only has '
                f'{usable_phase_count} usable phase slots; later phases will reuse slot '
                f'{usable_phase_count - 1}')

        phase_source_slots = {
            phase_idx: min(phase_idx, usable_phase_count - 1)
            for phase_idx in range(args.phase_count)
        }
        reserved_real_tokens = set()

        for phase_idx in range(args.phase_count):
            source_phase_slot = phase_source_slots[phase_idx]

            cam_bundle = OrderedDict(
                (cam_name, interval_frames_by_cam[cam_name][source_phase_slot])
                for cam_name in args.camera_names)
            raw_token = cam_bundle[args.token_camera]['token']
            use_real_token = (source_phase_slot == phase_idx)
            if (not use_real_token and
                    args.planned_export_phases is not None and
                    phase_idx in args.planned_export_phases and
                    source_phase_slot not in args.planned_export_phases and
                    raw_token not in reserved_real_tokens):
                use_real_token = True
            if use_real_token:
                export_token = raw_token
            else:
                export_token = build_synthetic_export_token(
                    token=raw_token,
                    phase_idx=phase_idx,
                    source_phase_slot=source_phase_slot,
                    keyframe_token=curr_keyframe_token)
            if (args.planned_export_phases is not None and
                    phase_idx in args.planned_export_phases and
                    export_token == raw_token):
                reserved_real_tokens.add(raw_token)

            frame_idx_in_phase = len(phase_infos[phase_idx])
            global_idx = interval_idx * args.frames_per_keyframe + phase_idx

            ts_stats = camera_bundle_timestamp_stats(cam_bundle)
            if ts_stats['spread'] > int(args.max_camera_spread_ms * 1000):
                print(
                    f'[WARN] Camera timestamp spread is large in scene {scene_name}, '
                    f'phase {phase_idx}, frame {frame_idx_in_phase}: '
                    f'{ts_stats["spread"]} us')

            min_lidar_idx = None
            if not args.allow_nonmonotonic_lidar:
                min_lidar_idx = last_lidar_indices[phase_idx]

            lidar_idx = find_nearest_lidar_index(
                lidar_timestamps=lidar_timestamps,
                anchor_timestamp=ts_stats['median'],
                min_idx=min_lidar_idx)
            ref_lidar_info = build_lidar_reference_info(nusc, lidar_chain[lidar_idx])

            lidar_dt = abs(ref_lidar_info['timestamp'] - ts_stats['median'])
            if lidar_dt > int(args.max_lidar_dt_ms * 1000):
                raise ValueError(
                    f'Lidar time gap too large in scene {scene_name}, '
                    f'phase {phase_idx}, frame {frame_idx_in_phase}: '
                    f'{lidar_dt} us')

            info = build_phase_info(
                nusc=nusc,
                scene_name=scene_name,
                scene_token=scene_token,
                phase_idx=phase_idx,
                frame_idx_in_phase=frame_idx_in_phase,
                frame_global_index=global_idx,
                cam_bundle=cam_bundle,
                ref_lidar_info=ref_lidar_info,
                keyframe_token=curr_keyframe_token,
                next_keyframe_token=next_keyframe_token,
                is_keyframe=(phase_idx == 0),
                token_camera=args.token_camera,
                source_phase_slot=source_phase_slot,
                usable_phase_count=usable_phase_count,
                export_token=export_token)
            phase_infos[phase_idx].append(info)
            last_lidar_indices[phase_idx] = lidar_idx

    last_keyframe_token = scene_keyframe_infos[-1]['token']
    last_sample = nusc.get('sample', last_keyframe_token)
    last_cam_bundle = OrderedDict(
        (cam_name, nusc.get('sample_data', last_sample['data'][cam_name]))
        for cam_name in args.camera_names)

    last_phase_idx = 0
    last_frame_idx_in_phase = len(phase_infos[last_phase_idx])
    last_global_idx = (len(scene_keyframe_infos) - 1) * args.frames_per_keyframe
    last_ts_stats = camera_bundle_timestamp_stats(last_cam_bundle)
    if last_ts_stats['spread'] > int(args.max_camera_spread_ms * 1000):
        print(
            f'[WARN] Camera timestamp spread is large in scene {scene_name}, '
            f'phase 0, frame {last_frame_idx_in_phase}: {last_ts_stats["spread"]} us')

    min_lidar_idx = None
    if not args.allow_nonmonotonic_lidar:
        min_lidar_idx = last_lidar_indices[last_phase_idx]
    lidar_idx = find_nearest_lidar_index(
        lidar_timestamps=lidar_timestamps,
        anchor_timestamp=last_ts_stats['median'],
        min_idx=min_lidar_idx)
    ref_lidar_info = build_lidar_reference_info(nusc, lidar_chain[lidar_idx])
    lidar_dt = abs(ref_lidar_info['timestamp'] - last_ts_stats['median'])
    if lidar_dt > int(args.max_lidar_dt_ms * 1000):
        raise ValueError(
            f'Lidar time gap too large in scene {scene_name}, '
            f'phase 0, frame {last_frame_idx_in_phase}: {lidar_dt} us')

    phase_infos[last_phase_idx].append(
        build_phase_info(
            nusc=nusc,
            scene_name=scene_name,
            scene_token=scene_token,
            phase_idx=last_phase_idx,
            frame_idx_in_phase=last_frame_idx_in_phase,
            frame_global_index=last_global_idx,
            cam_bundle=last_cam_bundle,
            ref_lidar_info=ref_lidar_info,
            keyframe_token=last_keyframe_token,
            next_keyframe_token=None,
            is_keyframe=True,
            token_camera=args.token_camera,
            source_phase_slot=0,
            usable_phase_count=1,
            export_token=last_cam_bundle[args.token_camera]['token']))

    return phase_infos


def dump_phase_pkls(phase_infos, metadata, out_dir, src_info):
    mkdir_or_exist(out_dir)
    stem = osp.splitext(osp.basename(src_info))[0]
    counts = {}

    for phase_idx, infos in phase_infos.items():
        out_path = osp.join(out_dir, f'{stem}_phase{phase_idx}.pkl')
        data = {
            'infos': infos,
            'metadata': metadata,
        }
        dump_pickle(data, out_path)
        counts[phase_idx] = len(infos)

    meta_path = osp.join(out_dir, 'meta.json')
    meta = {
        'source_info': src_info,
        'metadata': metadata,
        'phase_count': len(phase_infos),
        'frame_counts': counts,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if args.token_camera not in args.camera_names:
        raise ValueError('--token-camera must be contained in --camera-names')

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    base_infos, base_metadata = load_base_info(args.src_info)
    scene_groups = group_keyframe_infos_by_scene(nusc, base_infos)

    scene_items = list(scene_groups.items())
    if args.max_scenes is not None:
        scene_items = scene_items[:args.max_scenes]

    canonical_scene_groups = None
    canonical_metadata = {}
    history_keyframes = None
    steps_per_interval = None

    if args.canonical_info:
        canonical_scene_groups, canonical_metadata = load_canonical_info(args.canonical_info)
        canonical_example = None
        for scene_group in canonical_scene_groups.values():
            for info in scene_group['infos']:
                if len(info.get('frame_tokens', [])) > 0:
                    canonical_example = info
                    break
            if canonical_example is not None:
                break
        if canonical_example is None:
            raise ValueError(f'No canonical entry with frame_tokens found in {args.canonical_info}')
        history_keyframes, steps_per_interval = validate_canonical_settings(
            canonical_metadata=canonical_metadata,
            canonical_example=canonical_example,
            phase_labels=args.canonical_phase_labels)
        phase_infos = OrderedDict(
            (phase_label, []) for phase_label in args.canonical_phase_labels)
    else:
        if args.phase_count <= 0:
            raise ValueError('--phase-count must be positive')
        if args.frames_per_keyframe <= 0:
            raise ValueError('--frames-per-keyframe must be positive')
        if args.phase_count > args.frames_per_keyframe:
            raise ValueError('--phase-count should not exceed --frames-per-keyframe')
        if args.planned_export_phases is not None:
            invalid_phases = [
                phase_idx for phase_idx in args.planned_export_phases
                if phase_idx < 0 or phase_idx >= args.phase_count
            ]
            if invalid_phases:
                raise ValueError(
                    f'--planned-export-phases contains invalid indices: {invalid_phases}')
            args.planned_export_phases = sorted(set(args.planned_export_phases))
        phase_infos = {phase_idx: [] for phase_idx in range(args.phase_count)}

    processed_scenes = 0
    skipped_scenes = []

    for scene_token, scene_group in track_progress(scene_items):
        scene_name = scene_group['scene_name']
        scene_keyframe_infos = scene_group['infos']
        try:
            if args.canonical_info:
                canonical_scene_group = canonical_scene_groups.get(scene_token)
                if canonical_scene_group is None:
                    raise ValueError(
                        f'Scene {scene_name} missing in canonical info {args.canonical_info}')
                scene_phase_infos = process_one_scene_with_canonical(
                    nusc=nusc,
                    scene_token=scene_token,
                    scene_name=scene_name,
                    scene_keyframe_infos=scene_keyframe_infos,
                    scene_canonical_infos=canonical_scene_group['infos'],
                    args=args,
                    history_keyframes=history_keyframes,
                    steps_per_interval=steps_per_interval,
                    phase_labels=args.canonical_phase_labels)
            else:
                scene_phase_infos = process_one_scene(
                    nusc=nusc,
                    scene_token=scene_token,
                    scene_name=scene_name,
                    scene_keyframe_infos=scene_keyframe_infos,
                    args=args)
        except Exception as exc:
            if not args.skip_invalid_scenes:
                raise
            print(f'[WARN] Skip scene {scene_name}: {exc}')
            skipped_scenes.append({'scene_name': scene_name, 'error': str(exc)})
            continue

        for phase_idx in phase_infos:
            phase_infos[phase_idx].extend(scene_phase_infos[phase_idx])
        processed_scenes += 1

    metadata = dict(base_metadata)
    metadata.update({
        'source_info': args.src_info,
        'version': args.version,
        'camera_names': args.camera_names,
        'token_camera': args.token_camera,
        'max_camera_spread_ms': args.max_camera_spread_ms,
        'max_lidar_dt_ms': args.max_lidar_dt_ms,
        'processed_scenes': processed_scenes,
        'skipped_scenes': skipped_scenes,
    })
    if args.canonical_info:
        metadata.update({
            'generation_mode': 'canonical_semantic_phase_streams',
            'canonical_info': args.canonical_info,
            'canonical_phase_labels': args.canonical_phase_labels,
            'phase_count': len(args.canonical_phase_labels),
            'history_keyframes': history_keyframes,
            'steps_per_interval': steps_per_interval,
            'planned_export_phases': None,
        })
    else:
        metadata.update({
            'generation_mode': 'legacy_raw_slot_phase_streams',
            'phase_count': args.phase_count,
            'frames_per_keyframe': args.frames_per_keyframe,
            'planned_export_phases': args.planned_export_phases,
        })

    dump_phase_pkls(
        phase_infos=phase_infos,
        metadata=metadata,
        out_dir=args.out_dir,
        src_info=args.src_info)

    print(f'Processed scenes: {processed_scenes}')
    if skipped_scenes:
        print(f'Skipped scenes: {len(skipped_scenes)}')
    for phase_idx, infos in phase_infos.items():
        print(f'Phase {phase_idx}: {len(infos)} frames')
    print(f'Saved phase infos to {args.out_dir}')


if __name__ == '__main__':
    main()
