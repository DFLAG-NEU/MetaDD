from numpy.core.numeric import count_nonzero
import torch
from lreid.tools import time_now, CatMeter
from lreid.evaluation import (fast_evaluate_rank, fast_evaluate_rank_cuhksysu, compute_distance_matrix)
from IPython import embed
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from .visualising_rank import visualize_ranked_results
import numpy as np



def fast_test_p_s(config, base, loaders, current_step, if_test_forget=True):
    # using Cython test during train
    # return mAP, Rank-1
    base.set_all_model_eval()
    print(f'****** start perform fast testing! ******')
    # meters

    # compute query and gallery features
    def _cmc_map(_query_features_meter, _gallery_features_meter):
        query_features = _query_features_meter.get_val()
        gallery_features = _gallery_features_meter.get_val()


        distance_matrix = compute_distance_matrix(query_features, gallery_features, config.test_metric)
        distance_matrix = distance_matrix.data.cpu().numpy()
        CMC, mAP = fast_evaluate_rank(distance_matrix,
                                      query_pids_meter.get_val_numpy(),
                                      gallery_pids_meter.get_val_numpy(),
                                      query_cids_meter.get_val_numpy(),
                                      gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=False)

        return CMC[0] * 100, mAP * 100

    def _cmc_map_cuhk03(_query_features_meter, _gallery_features_meter):
        query_features = _query_features_meter.get_val()
        gallery_features = _gallery_features_meter.get_val()

        distance_matrix = compute_distance_matrix(query_features, gallery_features, config.test_metric)
        distance_matrix = distance_matrix.data.cpu().numpy()
        CMC, mAP = fast_evaluate_rank(distance_matrix,
                                      query_pids_meter.get_val_numpy(),
                                      gallery_pids_meter.get_val_numpy(),
                                      query_cids_meter.get_val_numpy(),
                                      gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=True,
                                      use_cython=False)

        return CMC[0] * 100, mAP * 100

    def _cmc_map_msmt17(_query_features_meter, _gallery_features_meter):
        query_features = _query_features_meter.get_val()
        gallery_features = _gallery_features_meter.get_val()
   
        # query -> devide 12 part -> [0, 1000] [1000, 2000] ... [11000, 11659]
        query_features_temp=[]
        query_pids_temp = []
        
        num = len(query_features)//11
        for i in range(0, len(query_features), num):
            query_features_temp.append(query_features[i:i+num])
            query_pids_temp.append(query_pids_meter.get_val_numpy()[i:i+num])

        CMC_temp = []
        mAP_temp = []
        for j in range(len(query_features_temp)):
            print('**************************', j)
            distance_matrix = compute_distance_matrix(query_features_temp[j], gallery_features, config.test_metric)
            distance_matrix = distance_matrix.data.cpu().numpy()
            CMC_t, mAP_t = fast_evaluate_rank(distance_matrix,
                                      query_pids_temp[j], 
                                      gallery_pids_meter.get_val_numpy(),
                                      query_pids_temp[j],
                                      gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=False)
            CMC_temp.append(CMC_t)
            mAP_temp.append(mAP_t)
        # embed()
        sum = 0.0
        for x in range(len(CMC_temp)):
            sum = sum + CMC_temp[x]
        CMC = sum/len(CMC_temp)
        # CMC = sum

        sum = 0.0
        for y in range(len(mAP_temp)):
            sum = sum+mAP_temp[y]
        mAP = sum/len(mAP_temp)
        
        return CMC[0] * 100, mAP * 100

    def _cmc_map_cuhksysu(_query_features_meter, _gallery_features_meter):
        query_features = _query_features_meter.get_val()
        gallery_features = _gallery_features_meter.get_val()


        distance_matrix = compute_distance_matrix(query_features, gallery_features, config.test_metric)
        distance_matrix = distance_matrix.data.cpu().numpy()
        CMC, mAP = fast_evaluate_rank_cuhksysu(distance_matrix,
                                      query_pids_meter.get_val_numpy(),
                                      gallery_pids_meter.get_val_numpy(),
                                      query_cids_meter.get_val_numpy(),
                                      gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=False)


        return CMC[0] * 100, mAP * 100

    results_dict = {}
    for dataset_name, temp_loaders in loaders.test_loader_dict.items():
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_metagraph_features_meter, query_metagraph_pids_meter, query_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_metagraph_features_meter, gallery_metagraph_pids_meter, gallery_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_fuse_features_meter, query_fuse_pids_meter, query_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_fuse_features_meter, gallery_fuse_pids_meter, gallery_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()
        # _datasets = CatMeter()
        
        print(time_now(), f' {dataset_name} feature start ')
        _datasets = []
        with torch.no_grad():
            for loader_id, loader in enumerate(temp_loaders): 
                _datasets.append(loader.sampler.data_source.samples)
                count = 0
                for data in loader:
                    count+=1
                    # compute feautres
                    images, pids, cids = data[0:3]
                    # print('test_p_s.py-------------line55')
                    # embed()
                    images = images.to(base.device)
                    features, featuremaps = base.model_dict['tasknet'](images, current_step)
                    if config.if_test_metagraph: 
                        features_metagraph, features_vk, _ = base.model_dict['metagraph'](features) #如果用到图的话，测试时将Vs在图中前传一次，得到Vs杠

                        features_fuse = features + features_metagraph 
                        # save as query features
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            query_fuse_features_meter.update(features_fuse.data)
                            query_metagraph_features_meter.update(features_metagraph.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    # save as gallery features
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            gallery_metagraph_features_meter.update(features_metagraph.data)
                            gallery_fuse_features_meter.update(features_fuse.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)


        print(time_now(), f' {dataset_name} feature done')
        if dataset_name=='subcuhksysu' or dataset_name=='cuhksysu':
            rank1, map = _cmc_map_cuhksysu(query_features_meter, gallery_features_meter)
            results_dict[f'{dataset_name}_tasknet_mAP'], results_dict[f'{dataset_name}_tasknet_Rank1'] = map, rank1
            if config.if_test_metagraph:
                rank1, map = _cmc_map_cuhksysu(query_fuse_features_meter, gallery_fuse_features_meter)
                results_dict[f'{dataset_name}_fuse_mAP'], results_dict[f'{dataset_name}_fuse_Rank1'] = map, rank1
        elif dataset_name=='msmt17':
            rank1, map = _cmc_map_msmt17(query_features_meter, gallery_features_meter)
            results_dict[f'{dataset_name}_tasknet_mAP'], results_dict[f'{dataset_name}_tasknet_Rank1'] = map, rank1
            if config.if_test_metagraph:
                rank1, map = _cmc_map_msmt17(query_fuse_features_meter, gallery_fuse_features_meter)
                results_dict[f'{dataset_name}_fuse_mAP'], results_dict[f'{dataset_name}_fuse_Rank1'] = map, rank1
        
        elif dataset_name=='cuhk03':
            rank1, map = _cmc_map_cuhk03(query_features_meter, gallery_features_meter)
            results_dict[f'{dataset_name}_tasknet_mAP'], results_dict[f'{dataset_name}_tasknet_Rank1'] = map, rank1
            if config.if_test_metagraph:   
                rank1, map = _cmc_map_cuhk03(query_fuse_features_meter, gallery_fuse_features_meter)
                results_dict[f'{dataset_name}_fuse_mAP'], results_dict[f'{dataset_name}_fuse_Rank1'] = map, rank1
        
        else:
            rank1, map = _cmc_map(query_features_meter, gallery_features_meter)
            results_dict[f'{dataset_name}_tasknet_mAP'], results_dict[f'{dataset_name}_tasknet_Rank1'] = map, rank1
            if config.if_test_metagraph:
                rank1, map = _cmc_map(query_fuse_features_meter, gallery_fuse_features_meter)
                results_dict[f'{dataset_name}_fuse_mAP'], results_dict[f'{dataset_name}_fuse_Rank1'] = map, rank1

    results_str = ''
    for criterion, value in results_dict.items():
        results_str = results_str + f'\n{criterion}: {value}'
    return results_dict, results_str


def save_and_fast_test_p_s(config, base, loaders, current_step, current_epoch,if_test_forget=True):
    # using Cython test during train
    # return mAP, Rank-1
    base.set_all_model_eval()
    print(f'****** start perform fast testing! ******')

    # meters

    # compute query and gallery features
    def _cmc_map(_query_features_meter, _gallery_features_meter):
        query_features = _query_features_meter.get_val()
        gallery_features = _gallery_features_meter.get_val()

        distance_matrix = compute_distance_matrix(query_features, gallery_features, config.test_metric)
        distance_matrix = distance_matrix.data.cpu().numpy()
        CMC, mAP = fast_evaluate_rank(distance_matrix,
                                      query_pids_meter.get_val_numpy(),
                                      gallery_pids_meter.get_val_numpy(),
                                      query_cids_meter.get_val_numpy(),
                                      gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=True)

        return CMC[0] * 100, mAP * 100

    results_dict = {}
    for dataset_name, temp_loaders in loaders.test_loader_dict.items():
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_metagraph_features_meter, query_metagraph_pids_meter, query_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_metagraph_features_meter, gallery_metagraph_pids_meter, gallery_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_fuse_features_meter, query_fuse_pids_meter, query_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_fuse_features_meter, gallery_fuse_pids_meter, gallery_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()

        print(time_now(), f' {dataset_name} feature start ')
        with torch.no_grad():
            for loader_id, loader in enumerate(temp_loaders):
                for data in loader:
                    # compute feautres
                    images, pids, cids = data[0:3]
                    images = images.to(base.device)
                    features, featuremaps = base.model_dict['tasknet'](images, current_step)
                    if config.if_test_metagraph:
                        # features_metagraph, _ = base.model_dict['metagraph'](features)
                        features_metagraph, protos_k, correlation = base.model_dict['metagraph'](features)
                        features_fuse = features + features_metagraph # 测试时用图， features+protos
                        # save as query features
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            query_fuse_features_meter.update(features_fuse.data)
                            query_metagraph_features_meter.update(features_metagraph.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    # save as gallery features
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            gallery_metagraph_features_meter.update(features_metagraph.data)
                            gallery_fuse_features_meter.update(features_fuse.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)

        print(time_now(), f' {dataset_name} feature done')
        rank1, map = _cmc_map(query_features_meter, gallery_features_meter)
        results_dict[f'{dataset_name}_tasknet_mAP'], results_dict[f'{dataset_name}_tasknet_Rank1'] = map, rank1
        if config.if_test_metagraph:
            rank1, map = _cmc_map(query_fuse_features_meter, gallery_fuse_features_meter)
            results_dict[f'{dataset_name}_fuse_mAP'], results_dict[f'{dataset_name}_fuse_Rank1'] = map, rank1

    results_str = ''
    for criterion, value in results_dict.items():
        results_str = results_str + f'\n{criterion}: {value}'
    return results_dict, results_str



