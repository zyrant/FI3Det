''' 
Author: Zhao Na
Data: September, 2020
https://github.com/Na-Z/SDCoT/blob/main/cfg/


Modify by zyrant
Data: 2025.6

'''

import argparse
import time
from os import path as osp
import os
import mmcv
import numpy as np
import random
from collections import defaultdict
random.seed(42)


scannet_class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                'garbagebin')

sunrgbd_class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

def create_scannet_infos(root_dir, out_dir, pkl_files, type_list, n_new_list, shot_list, mode='sort'):
    if mode == 'sort':
        print("Few-shot incremental classes are sorted by name")
        sorted_class_names = sorted(scannet_class_names)
    elif mode == 'random':
        sorted_class_names = list(scannet_class_names)
        random.shuffle(sorted_class_names)
    elif mode == 'direct':
        sorted_class_names = list(scannet_class_names)
    elif mode == 'count':
        '''
        chair: 4357
        door: 2026
        garbagebin: 1985
        cabinet: 1427
        table: 1271
        window: 928
        picture: 661
        desk: 551
        sofa: 406
        sink: 390
        bed: 307
        bookshelf: 300
        curtain: 292
        counter: 216
        toilet: 201
        refrigerator: 186
        showercurtrain: 116
        bathtub: 113
        '''
        # old_classes = {'chair','door','garbagebin','cabinet','table','window','picture','desk','sofa'}
        # new_classes = {'sink','bed','bookshelf','curtain','counter','toilet','refrigerator','showercurtrain','bathtub'}
        sorted_class_names = ['chair','door','garbagebin','cabinet','table','window','picture','desk','sofa',
                                'sink','bed','bookshelf','curtain','counter','toilet','refrigerator','showercurtrain','bathtub']
    else:
        raise NotImplementedError

    new_pkl = {}

    for n_new in n_new_list:
        for shot in shot_list:
            n_base = len(sorted_class_names) - int(n_new)
            base_names = sorted_class_names[:n_base]
            new_names = sorted_class_names[n_base:]
            
            print(f"\n>>> Base: {n_base}, New: {n_new}, Shot: {shot}")
            print(f"Base classes: {base_names}")
            print(f"New classes: {new_names}")

            for pkl_file in pkl_files:
                in_path = osp.join(root_dir, pkl_file)
                print(f"Reading: {in_path}")
                data_infos = mmcv.load(in_path)

                base_items = []
                selected_fewshot_items = []
                used_scene_ids_global = set()
                class_instance_counter = {cls: 0 for cls in new_names}
                max_shot = shot

                for item in data_infos:
                    if item['annos']['gt_num'] == 0:
                        # base_items.append(item.copy())
                        continue
                    else:
                        annos = item['annos']
                        obj_names = annos['name']
                        obj_classes = annos['class']

                        # base class
                        has_base = any(name in base_names for name in obj_names)

                        if has_base:
                            indices = [i for i, name in enumerate(obj_names) if name in base_names]
                            indices = np.array(indices)
                            new_annos = {
                                'gt_num': len(indices),
                                'name': obj_names[indices],
                                'class': obj_classes[indices],
                                'location': annos['location'][indices],
                                'dimensions': annos['dimensions'][indices],
                                'gt_boxes_upright_depth': annos['gt_boxes_upright_depth'][indices],
                                'unaligned_location': annos['unaligned_location'][indices],
                                'unaligned_dimensions': annos['unaligned_dimensions'][indices],
                                'unaligned_gt_boxes_upright_depth': annos['unaligned_gt_boxes_upright_depth'][indices],
                                'index': np.arange(len(indices), dtype=np.int32),
                                'axis_align_matrix': annos['axis_align_matrix'],
                            }
                            new_item = item.copy()
                            new_item['annos'] = new_annos
                            base_items.append(new_item)

    
                random.shuffle(data_infos)
                for item in data_infos:
                    scene_id = item['point_cloud']['lidar_idx']
                    if scene_id in used_scene_ids_global:
                        continue
                    
                    annos = item['annos']
                    if item['annos']['gt_num'] != 0:
                        obj_names = annos['name']
                        obj_classes = annos['class']

                        target_class_indices = {cls: [] for cls in new_names}
                        for i, name in enumerate(obj_names):
                            if name in new_names:
                                target_class_indices[name].append(i)

                        useful = any(len(target_class_indices[cls]) > 0 and class_instance_counter[cls] < max_shot for cls in new_names)
                        if not useful:
                            continue

                        keep_indices = []
                        for cls, idx_list in target_class_indices.items():
                            remain = max_shot - class_instance_counter[cls]
                            if remain <= 0:
                                continue
                            selected = random.sample(idx_list, min(remain, len(idx_list)))
                            class_instance_counter[cls] += len(selected)
                            keep_indices.extend(selected)

                        if not keep_indices:
                            continue

                        keep_indices = np.array(keep_indices)
                        new_annos = {
                            'gt_num': len(keep_indices),
                            'name': obj_names[keep_indices],
                            'class': obj_classes[keep_indices],
                            'location': annos['location'][keep_indices],
                            'dimensions': annos['dimensions'][keep_indices],
                            'gt_boxes_upright_depth': annos['gt_boxes_upright_depth'][keep_indices],
                            'unaligned_location': annos['unaligned_location'][keep_indices],
                            'unaligned_dimensions': annos['unaligned_dimensions'][keep_indices],
                            'unaligned_gt_boxes_upright_depth': annos['unaligned_gt_boxes_upright_depth'][keep_indices],
                            'index': np.arange(len(keep_indices), dtype=np.int32),
                            'axis_align_matrix': annos['axis_align_matrix'],
                        }

                        new_item = item.copy()
                        new_item['annos'] = new_annos
                        selected_fewshot_items.append(new_item)
                        used_scene_ids_global.add(scene_id)

                        if all(class_instance_counter[cls] >= max_shot for cls in new_names):
                            break

                from collections import defaultdict
                instance_count_summary = defaultdict(int)
                for item in selected_fewshot_items:
                    for name in item['annos']['name']:
                        if name in new_names:
                            instance_count_summary[name] += 1

                print(f"  Selected {len(selected_fewshot_items)} scenes")
                for cls in new_names:
                    print(f"  {cls:15s}: {instance_count_summary[cls]} instances")

                split_key = f'{n_new}_{shot}_increment'
                new_pkl[split_key] = {
                    f'{n_new}_old': base_items,
                    f'{n_new}_new': selected_fewshot_items
                }
            
    out_name = f'scannet_infos_train_few_shot_incremental_{mode}.pkl'
    out_path = osp.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    mmcv.dump(new_pkl, out_path)
    print(f"Written to: {out_path}")



def create_sunrgbd_infos(root_dir, out_dir, pkl_files, type_list, n_new_list, shot_list, mode='sort'):
    if mode == 'sort':
        print("Few-shot incremental classes are sorted by name")
        sorted_class_names = sorted(sunrgbd_class_names)
    elif mode == 'random':
        sorted_class_names = list(sunrgbd_class_names)
        random.shuffle(sorted_class_names)
    elif mode == 'direct':
        sorted_class_names = list(sunrgbd_class_names)
    elif mode == 'count':
        '''
        chair: 9278
        table: 2539
        desk: 933
        bed: 771
        sofa: 706
        night_stand: 293
        bookshelf: 204
        dresser: 182
        toilet: 171
        bathtub: 67
        '''
        sorted_class_names = ['chair','table','desk','bed','sofa',
                              'night_stand','bookshelf','dresser','toilet','bathtub']
    else:
        raise NotImplementedError


    new_pkl = {}

    for n_new in n_new_list:
        for shot in shot_list:
            n_base = len(sorted_class_names) - int(n_new)
            base_names = sorted_class_names[:n_base]
            new_names = sorted_class_names[n_base:]
            

            print(f"\n>>> Base: {n_base}, New: {n_new}, Shot: {shot}")
            print(f"Base classes: {base_names}")
            print(f"New classes: {new_names}")

            for pkl_file in pkl_files:
                in_path = osp.join(root_dir, pkl_file)
                print(f"Reading: {in_path}")
                data_infos = mmcv.load(in_path)

                base_items = []
                selected_fewshot_items = []
                used_scene_ids_global = set()
                class_instance_counter = {cls: 0 for cls in new_names}
                max_shot = shot

                for item in data_infos:
                    if item['annos']['gt_num'] == 0:
                        # base_items.append(item.copy())
                        continue
                    else:
                    
                        annos = item['annos']
                        obj_names = annos['name']
                        obj_classes = annos['class']

                        has_base = any(name in base_names for name in obj_names)
                        if has_base:
                            indices = [i for i, name in enumerate(obj_names) if name in base_names]
                            if not indices:
                                continue
                            indices = np.array(indices)
                            new_annos = {
                                'gt_num': len(indices),
                                'name' : obj_names[indices],
                                'bbox' : annos['bbox'][indices],
                                'location' : annos['location'][indices],
                                'dimensions' : annos['dimensions'][indices],  # lwh (depth) format
                                'rotation_y' : annos['rotation_y'][indices],
                                'index' : np.arange(len(indices), dtype=np.int32),
                                'class' : obj_classes[indices],
                                'gt_boxes_upright_depth' : annos['gt_boxes_upright_depth'][indices],  # (K,8)
                            }
                            new_item = item.copy()
                            new_item['annos'] = new_annos
                            base_items.append(new_item)

                      
                random.shuffle(data_infos)
                for item in data_infos:
                    scene_id = item['point_cloud']['lidar_idx']
                    if scene_id in used_scene_ids_global:
                        continue
                    
                    annos = item['annos']
                    if item['annos']['gt_num'] == 0:
                        continue
            
                    obj_names = annos['name']
                    obj_classes = annos['class']

                    target_class_indices = {cls: [] for cls in new_names}
                    for i, name in enumerate(obj_names):
                        if name in new_names:
                            target_class_indices[name].append(i)

                    useful = any(len(target_class_indices[cls]) > 0 and class_instance_counter[cls] < max_shot for cls in new_names)
                    if not useful:
                        continue

                    keep_indices = []
                    for cls, idx_list in target_class_indices.items():
                        remain = max_shot - class_instance_counter[cls]
                        if remain <= 0:
                            continue
                        selected = random.sample(idx_list, min(remain, len(idx_list)))
                        class_instance_counter[cls] += len(selected)
                        keep_indices.extend(selected)

                    if not keep_indices:
                        continue

                    keep_indices = np.array(keep_indices)
                    new_annos = {
                        'gt_num': len(keep_indices),
                        'name' : obj_names[keep_indices],
                        'bbox' : annos['bbox'][keep_indices],
                        'location' : annos['location'][keep_indices],
                        'dimensions' : annos['dimensions'][keep_indices],  # lwh (depth) format
                        'rotation_y' : annos['rotation_y'][keep_indices],
                        'index' : np.arange(len(keep_indices), dtype=np.int32),
                        'class' : obj_classes[keep_indices],
                        'gt_boxes_upright_depth' : annos['gt_boxes_upright_depth'][keep_indices],  # (K,8)
                    }

                    new_item = item.copy()
                    new_item['annos'] = new_annos
                    selected_fewshot_items.append(new_item)
                    used_scene_ids_global.add(scene_id)

                    if all(class_instance_counter[cls] >= max_shot for cls in new_names):
                        break

                from collections import defaultdict
                instance_count_summary = defaultdict(int)
                for item in selected_fewshot_items:
                    for name in item['annos']['name']:
                        if name in new_names:
                            instance_count_summary[name] += 1

                print(f"  Selected {len(selected_fewshot_items)} scenes")
                for cls in new_names:
                    print(f"  {cls:15s}: {instance_count_summary[cls]} instances")

                split_key = f'{n_new}_{shot}_increment'
                new_pkl[split_key] = {
                    f'{n_new}_old': base_items,
                    f'{n_new}_new': selected_fewshot_items
                }
            
    out_name = f'sunrgbd_infos_train_few_shot_incremental_{mode}.pkl'
    out_path = osp.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    mmcv.dump(new_pkl, out_path)
    print(f"Written to: {out_path}")


parser = argparse.ArgumentParser(description='Create few_increment annos.')
parser.add_argument('--dataset', default='sunrgbd', help='name of the dataset')
parser.add_argument(
    '--root-dir',
    type=str,
    default='/opt/data/private/all_data/sunrgbd/',
    help='specify the root dir of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='/opt/data/private/all_data/sunrgbd/few_shot_increment/',
    help='specify the out dir of dataset')
parser.add_argument('--type', default=['batch',], help='type of incremental')
parser.add_argument('--n_new', default=[5, 1], help='number of incremental class') #scannet 9,1  sunrgbd 5,1
parser.add_argument('--shot', default=[1, 5], help='labeled number in incremental class') # fix 1,5
parser.add_argument('--mode', default='count', help='creat mode')



args = parser.parse_args()


if __name__ == '__main__':
    if args.out_dir is None:
        args.out_dir = args.root_dir
    elif args.dataset == 'scannet_v2':
        pkl_files = ['scannet_infos_train.pkl']
        create_scannet_infos(
            root_dir=args.root_dir, 
            out_dir=args.out_dir, 
            pkl_files=pkl_files,
            type_list=args.type, 
            n_new_list = args.n_new,
            shot_list = args.shot,
            mode = args.mode,
            )
    elif args.dataset == 'sunrgbd':
        pkl_files = ['sunrgbd_infos_train.pkl']
        create_sunrgbd_infos(
            root_dir=args.root_dir, 
            out_dir=args.out_dir, 
            pkl_files=pkl_files,
            type_list=args.type, 
            n_new_list = args.n_new,
            shot_list = args.shot,
            mode = args.mode,
            )

