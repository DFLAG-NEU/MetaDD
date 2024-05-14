import torch
from lreid.tools import MultiItemAverageMeter
from lreid.evaluation import accuracy
from IPython import embed
from collections import OrderedDict
import copy
from torchvision.transforms import transforms
import numpy as np
import random

'''
Continual Adaptation for Visual Representation
'''


def random_transform(imgs, flag):
    if flag == 0:
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif flag==1: 
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            # transforms.RandomHorizontalFlip(p=0.5)
            transforms.RandomRotation(degrees=30), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif flag==2: # color + degree + gauss
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            # transforms.RandomHorizontalFlip(p=0.5)
            transforms.RandomRotation(degrees=30),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))
            # RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]) 
        ]
    elif flag==3: # init
        transform_train = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    transform_train =transforms.Compose(transform_train)
    
    imgs_aug = torch.Tensor(np.zeros_like(imgs.cpu()))
    for i in range(len(imgs)):
        imgs_aug[i] = transform_train(imgs[i])

    return imgs_aug

def random_transform_2(imgs, flag):
    if flag == 0:
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif flag==1: 
        transform_train = [
            transforms.ToPILImage(),
            # transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            # transforms.RandomHorizontalFlip(p=0.5)
            transforms.RandomRotation(degrees=30), # 随机旋转30度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif flag==2: 
        transform_train = [
            # transforms.Resize(config.image_size, interpolation=3),
            transforms.ToPILImage(),
            # transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            # transforms.RandomHorizontalFlip(p=0.5)
            # transforms.RandomRotation(degrees=30),  # 随机旋转30度
            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),           
            # RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]) 
        ]
    elif flag==3: 
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomRotation(degrees=30), # 随机旋转30度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif flag==4: 
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif flag==5: 
        transform_train = [
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=30), # 随机旋转30度
            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    elif flag==6: 
        transform_train = [
            # transforms.Resize(config.image_size, interpolation=3),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            # transforms.RandomHorizontalFlip(p=0.5)
            transforms.RandomRotation(degrees=30),  # 随机旋转30度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))
            # RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]) 
        ]
    else:
        transform_train = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    transform_train =transforms.Compose(transform_train)
    
    imgs_aug = torch.Tensor(np.zeros_like(imgs.cpu()))
    for i in range(len(imgs)):
        imgs_aug[i] = transform_train(imgs[i])

    return imgs_aug


def train_p_s_an_epoch(config, base, loader, current_step, old_model, old_graph_model, current_epoch=None, output_featuremaps=True):
    base.set_all_model_train()
    meter = MultiItemAverageMeter()
    if old_model is None:
        print('****** training tasknet ******\n')
    else:
        print('****** training both tasknet and metagraph ******\n')
    heatmaps_dict = {}
    for _ in range(config.steps):
        base.set_model_and_optimizer_zero_grad()
        ### load a batch data
        mini_batch = loader.continual_train_iter_dict[current_step].next_one()

        while mini_batch[0].size(0) != 2* config.p * config.k:
            mini_batch = loader.continual_train_iter_dict[current_step].next_one()
        
        
        if len(mini_batch) > 6:
            assert config.continual_step == 'task'

        imgs, global_pids, global_cids, dataset_name, local_pids, image_paths = mini_batch
        

        l_img, l_pid = len(imgs)//2, len(local_pids)//2
        imgs_1, local_pids_1, = imgs[:l_img], local_pids[:l_pid]
        # imgs_2 for meta-train
        imgs_2, local_pids_2, = imgs[l_img:], local_pids[l_pid:]   
        imgs_1, local_pids_1 = imgs_1.to(base.device), local_pids_1.to(base.device)
        imgs_2, local_pids_2 = imgs_2.to(base.device), local_pids_2.to(base.device)
        imgs, local_pids = imgs_2, local_pids_2
        imgs, local_pids = imgs.to(base.device), local_pids.to(base.device)      
        flag = random.choice([0,1,2])
        imgs_aug_1 = random_transform(imgs_1, flag)      
        local_pids_aug_1 = local_pids_1

        imgs_aug, local_pids_aug = imgs_aug_1.to(base.device), local_pids_aug_1.to(base.device)
        del mini_batch, imgs_1, imgs_2
        torch.cuda.empty_cache()

        loss = 0
        loss_aug = 0 
        loss_b1 = 0
        loss_b2 = 0

        meta_lr = 0.1 

        if old_model is None:  
            features, cls_score, feature_maps = base.model_dict['tasknet'](imgs, current_step)
            feature_temp = features

            features_a, cls_score_a, feature_maps_a = base.model_dict['tasknet'](imgs_aug, current_step)
            
            features = torch.cat((features, features_a), 0)  
            protos, protos_a, protos_k, _ = base.model_dict['metagraph'](features.detach())
            
            feature_fuse_b1 = feature_temp + protos 
            
            feature_fuse = features_a + protos_a 
            

            ide_loss_b1 = config.weight_x * base.ide_criterion(cls_score, local_pids)
            plasticity_loss_b1 = config.weight_t * base.triplet_criterion(feature_fuse_b1, feature_fuse_b1, feature_fuse_b1, local_pids, local_pids, local_pids)
            
            loss_b1 += ide_loss_b1
            loss_b1 += plasticity_loss_b1
            
            ide_loss = config.weight_x * base.ide_criterion(cls_score_a, local_pids_aug) 
            plasticity_loss = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse, local_pids_aug, local_pids_aug, local_pids_aug)
            
            loss += ide_loss
            loss += plasticity_loss

            grads = torch.autograd.grad(loss, base.model_dict['tasknet'].parameters(), retain_graph=True, allow_unused=True)
            tasknet_tmp = copy.deepcopy(base.model_dict['tasknet'])

            for param_tmp, param, grad in zip(tasknet_tmp.parameters(), base.model_dict['tasknet'].parameters(), grads):
                if grad is not None:
                    param_tmp.data.copy_((param_tmp.data -meta_lr * grad).data)
                    param_tmp.grad = grad
 

            features_2, cls_score_2, feature_maps_2  = tasknet_tmp(imgs, current_step)
            features_aug, cls_score_aug, feature_maps_aug  = tasknet_tmp(imgs_aug, current_step)
            
            features_fuse_b2 = features_2 + protos
            features_aug_new = features_aug + protos_a
            ide_loss_b2 = config.weight_x * base.ide_criterion(cls_score_2, local_pids)
            plasticity_loss_b2 = config.weight_t * base.triplet_criterion(features_fuse_b2, features_fuse_b2, features_fuse_b2, local_pids, local_pids, local_pids)   
            
            ide_loss_aug = config.weight_x * base.ide_criterion(cls_score_aug, local_pids_aug)                            
            plasticity_loss_aug = config.weight_t * base.triplet_criterion(features_aug_new, features_aug_new, features_aug_new, local_pids_aug, local_pids_aug, local_pids_aug)   
          
            loss_b2 += ide_loss_b2 
            loss_b2 += plasticity_loss_b2

            loss_aug += ide_loss_aug     
            loss_aug += plasticity_loss_aug


            meter.update({
                'ide_loss_b1': ide_loss_b1.data,
                'ide_loss_b2': ide_loss_b2.data,
                'ide_loss_aug':ide_loss_aug.data,
                'plasticity_loss_b1': plasticity_loss_b1.data,
                'plasticity_loss_b2': plasticity_loss_b2.data,
                'plasticity_loss_aug': plasticity_loss_aug.data,
            })

        else:
            old_current_step = list(range(current_step))
            new_current_step = list(range(current_step + 1))
            features, cls_score_list, feature_maps = base.model_dict['tasknet'](imgs, new_current_step)
            cls_score = cls_score_list[-1]
            feature_temp = features

            features_a, cls_score_list_a, feature_maps_a = base.model_dict['tasknet'](imgs_aug, new_current_step)
            cls_score_a = cls_score_list_a[-1]

            features = torch.cat((features, features_a), 0) 
            protos, protos_a, protos_k, correlation = base.model_dict['metagraph'](features.detach())
            
            feature_fuse_b1 = feature_temp + protos
            feature_fuse = features_a + protos_a

            ide_loss_b1 = config.weight_x * base.ide_criterion(cls_score, local_pids)
            plasticity_loss_b1 = config.weight_t * base.triplet_criterion(feature_fuse_b1, feature_fuse_b1, feature_fuse_b1,
                                                                    local_pids, local_pids, local_pids)
            loss_b1 += ide_loss_b1
            loss_b1 += plasticity_loss_b1

            ide_loss = config.weight_x * base.ide_criterion(cls_score_a, local_pids_aug)
            plasticity_loss = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse,
                                                                    local_pids_aug, local_pids_aug, local_pids_aug)
             
            loss +=ide_loss
            loss += plasticity_loss

           

            with torch.no_grad():
                old_features, old_cls_score_list, old_feature_maps = old_model(imgs, old_current_step)
                old_features_aug, old_cls_score_list_aug, old_feature_maps_aug = old_model(imgs_aug, old_current_step)
                old_vertex = old_graph_model.meta_graph_vertex

            
            # for fast weights
            new_logit = torch.cat(cls_score_list_a, dim=1)
            old_logit = torch.cat(old_cls_score_list_aug, dim=1) 

            knowladge_distilation_loss = config.weight_kd * base.loss_fn_kd(new_logit, old_logit, config.kd_T)                    
            loss += knowladge_distilation_loss

            
             # for total_loss first part
            new_logit_b1 = torch.cat(cls_score_list, dim=1)
            old_logit_b1 = torch.cat(old_cls_score_list, dim=1) 
            
            knowladge_distilation_loss_b1 = config.weight_kd * base.loss_fn_kd(new_logit_b1, old_logit_b1, config.kd_T)  
            loss_b1 += knowladge_distilation_loss_b1
                
            grads = torch.autograd.grad(loss, base.model_dict['tasknet'].parameters(), retain_graph=True, allow_unused=True)

            tasknet_tmp = copy.deepcopy(base.model_dict['tasknet'])
            # step1: use raw image for updating
            for param_tmp, param, grad in zip(tasknet_tmp.parameters(), base.model_dict['tasknet'].parameters(), grads):
                if grad is not None:
                    param_tmp.data.copy_((param_tmp.data - meta_lr * grad).data)
                    param_tmp.grad = grad
            
            # with torch.no_grad():
            features_2, cls_score_list_2, feature_maps_2 = tasknet_tmp(imgs, new_current_step)
            cls_score_2 = cls_score_list_2[-1]
            
            features_aug, cls_score_list_aug, feature_maps_aug = tasknet_tmp(imgs_aug, new_current_step)                      
            cls_score_aug = cls_score_list_aug[-1]

            new_logit_b2 = torch.cat(cls_score_list_2, dim=1)
            new_logit_aug = torch.cat(cls_score_list_aug, dim=1)

            knowladge_distilation_loss_b2 = config.weight_kd * base.loss_fn_kd(new_logit_b2, old_logit_b1, config.kd_T)    
            loss_b2 +=knowladge_distilation_loss_b2
            
            knowladge_distilation_loss_aug = config.weight_kd * base.loss_fn_kd(new_logit_aug, old_logit, config.kd_T)                    
            loss_aug += knowladge_distilation_loss_aug

            feature_fuse = features_aug + protos_a
            feature_fuse_b2 = features_2 + protos
            
            ide_loss_b2 = config.weight_x * base.ide_criterion(cls_score_2, local_pids)
            
            ide_loss_aug = config.weight_x * base.ide_criterion(cls_score_aug, local_pids_aug)
            
            plasticity_loss_b2 = config.weight_t * base.triplet_criterion(feature_fuse_b2, feature_fuse_b2, feature_fuse_b2,
                                                                    local_pids, local_pids, local_pids)
            
            plasticity_loss_aug = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse,
                                                                    local_pids_aug, local_pids_aug, local_pids_aug)
            
            stability_loss = config.weight_r * base.model_dict['metagraph'].StabilityLoss(old_vertex, base.model_dict['metagraph'].meta_graph_vertex)
            
            loss_aug += ide_loss_aug
            loss_aug += plasticity_loss_aug

            loss_b2 += ide_loss_b2
            loss_b2 += plasticity_loss_b2
               
            loss_aug += stability_loss

            meter.update({
                'ide_loss_b1': ide_loss_b1.data,
                'ide_loss_b2': ide_loss_b2.data,
                'ide_loss_aug':ide_loss_aug.data,
                'plasticity_loss_b1': plasticity_loss_b1.data,
                'plasticity_loss_b2': plasticity_loss_b2.data,
                'plasticity_loss_aug': plasticity_loss_aug.data,
                'stability_loss': stability_loss.data,
            })



        total_loss = loss_b1 + loss_b2 + loss_aug 

        ### optimize
        base.optimizer_dict['tasknet'].zero_grad()
        base.optimizer_dict['metagraph'].zero_grad()
        if config.fp_16:  # we use optimier to backward loss
            with base.amp.scale_loss(total_loss, base.optimizer_list) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
            

        base.optimizer_dict['tasknet'].step() 
        base.optimizer_dict['metagraph'].step() 

    if config.re_init_lr_scheduler_per_step:
        _lr_scheduler_step = current_epoch
    else:
        _lr_scheduler_step = current_step * config.total_train_epochs + current_epoch
    base.lr_scheduler_dict['tasknet'].step(_lr_scheduler_step)
    base.lr_scheduler_dict['metagraph'].step(_lr_scheduler_step)

    if output_featuremaps and not config.output_featuremaps_from_fixed:
        heatmaps_dict['feature_maps_true'] = base.featuremaps2heatmaps(imgs.detach().cpu(), feature_maps.detach().cpu(),
                                                                       image_paths,
                                                                       current_epoch,
                                                                       if_save=config.save_heatmaps,
                                                                       if_fixed=False,
                                                                       if_fake=False
                                                                       )
        return (meter.get_value_dict(), meter.get_str(), heatmaps_dict)
    else:
        return (meter.get_value_dict(), meter.get_str())