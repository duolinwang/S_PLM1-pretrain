import torch
import torch.nn.functional as F
import os
import yaml
import shutil
import numpy as np
import logging as log
from scipy.ndimage import zoom
from box import Box
from pathlib import Path
import datetime
from timm import optim
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts



def prepare_tensorboard(result_path):
    train_log_path = os.path.join(result_path, 'tensorboard')
    Path(train_log_path).mkdir(parents=True, exist_ok=True)
    train_writer = SummaryWriter(train_log_path)

    return train_writer


def get_logging(result_path):
    logger = log.getLogger(result_path)
    logger.setLevel(log.INFO)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger

def save_config_file(model_checkpoints_folder, args): 
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def prepare_saving_dir(configs,config_file_path):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Add '_evaluation' to the run_id if the 'evaluate' flag is True.
    # if configs.evaluate:
    #     run_id += '_evaluation'

    # Create the result directory and the checkpoint subdirectory.
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    #I have to remobe the run_id for easier analyze on results.
    #result_path = os.path.abspath(configs.result_path)
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    figures_path = os.path.join(result_path, 'figures')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(figures_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    #shutil.copy('config.yaml', result_path)
    #save_config_file(result_path,configs.to_dict())
    shutil.copy(config_file_path, os.path.join(result_path,'config.yaml'))
    # Return the path to the result directory.
    return result_path, checkpoint_path


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def load_configs(config,args):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)
    
    # Convert the necessary values to floats.
    tree_config.optimizer.lr_seq = float(tree_config.optimizer.lr_seq)
    tree_config.optimizer.lr_struct = float(tree_config.optimizer.lr_struct)
    tree_config.optimizer.decay.min_lr_struct = float(tree_config.optimizer.decay.min_lr_struct)
    tree_config.optimizer.decay.min_lr_seq = float(tree_config.optimizer.decay.min_lr_seq)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    tree_config.train_settings.contactmethod.dist1 = float(tree_config.train_settings.contactmethod.dist1)
    tree_config.train_settings.contactmethod.dist2 = float(tree_config.train_settings.contactmethod.dist2)
    tree_config.train_settings.contactmethod.dist3 = float(tree_config.train_settings.contactmethod.dist3)
    tree_config.train_settings.temperature = float(tree_config.train_settings.temperature)
    tree_config.optimizer.beta_1  = float(tree_config.optimizer.beta_1)
    tree_config.optimizer.beta_2 = float(tree_config.optimizer.beta_2)
    tree_config.model.esm_encoder.lora.dropout = float(tree_config.model.esm_encoder.lora.dropout)
    tree_config.model.swin_encoder.lora.dropout = float(tree_config.model.swin_encoder.lora.dropout)
    
    #overwrite parameters if set through commandline
    if args is not None:
        if args.result_path:
           tree_config.result_path=args.result_path
        
        if args.resume_path:
           tree_config.resume.resume_path = args.resume_path
        
        if args.num_end_adapter_layers:
           tree_config.encoder.adapter_h.num_end_adapter_layers=int(args.num_end_adapter_layers)
        
        if args.module_type:
           tree_config.encoder.adapter_h.module_type=args.module_type
    
    
    return tree_config


def load_checkpoints(simclr, configs,logging):
    if configs.resume.resume:
        if configs.resume.resume_path is not None:
            checkpoint = torch.load(configs.resume.resume_path)#, map_location=lambda storage, loc: storage)
            print(f"load checkpoints from {configs.resume.resume_path}")
            logging.info(f"load checkpoints from {configs.resume.resume_path}")
        else:
            checkpoint = torch.load('checkpoint_initial.pth.tar')#, map_location=lambda storage, loc: storage)
            print(f"load checkpoints from checkpoint_initial.pth.tar")
            logging.info(f"load checkpoints from checkpoint_initial.pth.tar")
        
        simclr.model_struct.load_state_dict(checkpoint['state_dict2'])
        #to load old checkpoints that saved adapter_layer_dict as adapter_layer. 
        from collections import OrderedDict
        if np.sum(["adapter_layer_dict" in key for key in checkpoint['state_dict1'].keys()])==0: #using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
             new_ordered_dict = OrderedDict()
             for key, value in checkpoint['state_dict1'].items():
                 if "adapter_layer_dict" not in key:
                   new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                   new_ordered_dict[new_key] = value
                 else:
                   new_ordered_dict[key] = value
             
             simclr.model_seq.load_state_dict(new_ordered_dict)
        else: #new checkpoints with new code, that can be loaded directly.
           simclr.model_seq.load_state_dict(checkpoint['state_dict1'])
        
        if hasattr(configs.model, 'memory_banck'):
          if configs.model.memory_banck.enable:
            #"""
            if "seq_queue" in checkpoint:
                simclr.register_buffer('seq_queue', checkpoint['seq_queue'])
                simclr.register_buffer('seq_queue_ptr', checkpoint['seq_queue_ptr'])
                simclr.register_buffer('struct_queue', checkpoint['struct_queue'])
                simclr.register_buffer('struct_queue_ptr', checkpoint['struct_queue_ptr'])
                #simclr.seq_queue=simclr.seq_queue.to('cpu')
                #simclr.seq_queue_ptr=simclr.seq_queue_ptr.to('cpu')
                #simclr.struct_queue=simclr.struct_queue.to('cpu')
                #simclr.struct_queue_ptr=simclr.struct_queue_ptr.to('cpu')
                """
                simclr.seq_queue.load_state_dict(checkpoint['seq_queue'])
                simclr.seq_queue_ptr.load_state_dict(checkpoint['seq_queue_ptr'])
                simclr.struct_queue.load_state_dict(checkpoint['struct_queue'])
                simclr.struct_queue_ptr.load_state_dict(checkpoint['struct_queue_ptr'])
                """
    
    return simclr


def save_checkpoints(optimizer_struct, optimizer_seq, result_path, simclr, n_steps, logging, epoch,enable_queue=False):
    checkpoint_name = f'checkpoint_{n_steps:07d}.pth'
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)

    save_checkpoint({
        'epoch': epoch,
        'step': n_steps,
        'state_dict1': simclr.model_seq.state_dict(),
        'state_dict2': simclr.model_struct.state_dict(),
        'optimizer_struct': optimizer_struct.state_dict(),
        'optimizer_seq': optimizer_seq.state_dict(),
        'seq_queue':simclr.seq_queue if enable_queue else {},
        'seq_queue_ptr':simclr.seq_queue_ptr if enable_queue else {},
        'struct_queue': simclr.struct_queue if enable_queue else {},
        'struct_queue_ptr':simclr.struct_queue_ptr if enable_queue else {},
    }, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata have been saved at {save_path}")


def load_optimizers(model_seq, model_struct, logging, configs):
    if configs.optimizer.name.lower() == 'adabelief':
        optimizer_seq = optim.AdaBelief(model_seq.parameters(), lr=configs.optimizer.lr_seq, eps=configs.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=configs.optimizer.weight_decay, rectify=False)
        optimizer_struct = optim.AdaBelief(model_struct.parameters(), lr=configs.optimizer.lr_struct, eps=configs.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=configs.optimizer.weight_decay, rectify=False)
    elif configs.optimizer.name.lower() == 'adam':
        if configs.optimizer.use_8bit_adam:
            import bitsandbytes
            logging.info('use 8-bit adamw')
            optimizer_seq = bitsandbytes.optim.AdamW8bit(
                model_seq.parameters(), lr=float(configs.optimizer.lr_seq),
                betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                weight_decay=float(configs.optimizer.weight_decay),
                eps=float(configs.optimizer.eps),
            )
            optimizer_struct = bitsandbytes.optim.AdamW8bit(
                model_struct.parameters(), lr=float(configs.optimizer.lr_struct),
                betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                weight_decay=float(configs.optimizer.weight_decay),
                eps=float(configs.optimizer.eps),
            )
        else:
            optimizer_seq = torch.optim.AdamW(
                model_seq.parameters(), lr=float(configs.optimizer.lr_seq),
                betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                weight_decay=float(configs.optimizer.weight_decay),
                eps=float(configs.optimizer.eps)
            )
            optimizer_struct = torch.optim.AdamW(
                model_struct.parameters(), lr=float(configs.optimizer.lr_struct),
                betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                weight_decay=float(configs.optimizer.weight_decay),
                eps=float(configs.optimizer.eps)
            )
    elif configs.optimizer.name.lower() == 'sgd':
             logging.info('use sgd optimizer')
             optimizer_struct = torch.optim.SGD(model_struct.parameters(), lr=float(configs.optimizer.lr_struct), momentum=0.9, dampening=0, weight_decay=float(configs.optimizer.weight_decay))
             optimizer_seq = torch.optim.SGD(model_seq.parameters(), lr=float(configs.optimizer.lr_seq), momentum=0.9, dampening=0, weight_decay=float(configs.optimizer.weight_decay))
    
    else:
        raise ValueError('wrong optimizer')
    return optimizer_seq, optimizer_struct


def prepare_optimizer(model_seq, model_struct, logging, configs):
    logging.info("prepare the optimizers")
    optimizer_seq, optimizer_struct = load_optimizers(model_seq, model_struct, logging, configs)

    print(optimizer_seq.param_groups[0].keys())
    print(optimizer_struct.param_groups[0].keys())

    logging.info("prepare the schedulers")
    #if configs.optimizer.name.lower() != 'sgd':
    scheduler_struct = CosineAnnealingWarmupRestarts(
       optimizer_struct,
       first_cycle_steps=configs.optimizer.decay.first_cycle_steps,
       cycle_mult=1.0,
       max_lr=configs.optimizer.lr_struct,
       min_lr=configs.optimizer.decay.min_lr_struct,
       warmup_steps=configs.optimizer.decay.warmup,
       gamma=configs.optimizer.decay.gamma)
    
    #plot_learning_rate(scheduler=scheduler_struct, optimizer=optimizer_struct, num_steps=10000,filename="LR_"+str(configs.optimizer.decay.gamma)+".png")#configs.epochs)
    #exit()
    scheduler_seq = CosineAnnealingWarmupRestarts(
       optimizer_seq,
       first_cycle_steps=configs.optimizer.decay.first_cycle_steps,
       cycle_mult=1.0,
       max_lr=configs.optimizer.lr_seq,
       min_lr=configs.optimizer.decay.min_lr_seq,
       warmup_steps=configs.optimizer.decay.warmup,
       gamma=configs.optimizer.decay.gamma)
    #else:
    #    logging.info("use sgd and old lr_scheduler")
    #    scheduler_struct = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_struct, configs.optimizer.decay.T0, configs.optimizer.decay.Tmult)
    #    scheduler_seq = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_seq, configs.optimizer.decay.T0, configs.optimizer.decay.Tmult)
    
    
    return scheduler_seq, scheduler_struct, optimizer_seq, optimizer_struct


def plot_learning_rate(scheduler, optimizer, num_steps,filename):
    import matplotlib.pyplot as plt
    # Collect learning rates
    lrs = []
    
    # Simulate a training loop
    for epoch in range(num_steps):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    # Plot the learning rates
    plt.plot(lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    #plt.show()
    plt.savefig(filename)


def Image_resize(matrix, max_len):
    bsz = len(matrix)
    new_size = [max_len, max_len]
    pad_contact = np.zeros((bsz, max_len, max_len), dtype=float)
    for index in range(bsz):
        scale_factors = (new_size[0] / matrix[index].shape[0], new_size[1] / matrix[index].shape[1])
        pad_contact[index] = zoom(matrix[index], zoom=scale_factors, order=0)  # order=0 for nearest-neighbor, order=3 for bicubic order=1 to perform bilinear interpolation, which smoothly scales the matrix content to the desired size.

    return pad_contact


def pad_concatmap(matrix, max_len, pad_value=255):
    bsz = len(matrix)
    pad_contact = np.zeros((bsz, max_len, max_len), dtype=float)
    mask_matrix = np.full((bsz, max_len, max_len), False, dtype=bool)
    for i in range(bsz):
        leng = len(matrix[i])
        if leng >= max_len:
            # print(i)
            pad_contact[i, :, :] = matrix[i][:max_len,
                                   :max_len]  # we trim the contact map 2D matrix if it's dimension > max_len
            mask_matrix[i, :, :] = True
        else:
            # print(i)
            # print("len < 224")
            pad_len = max_len - leng
            pad_contact[i, :, :] = np.pad(matrix[i], [(0, pad_len), (0, pad_len)], mode='constant',
                                          constant_values=pad_value)
            mask_matrix[i, :leng, :leng] = True
            # print(mask_matrix[i].shape)
            # print(mask_matrix[i])

    return pad_contact, mask_matrix


# The first method to convert contact map
def contact_3Dchannel_v1(images, dis1=10.0, dis2=20.0, dis3=30.0):  # revise the thredshold to 10 20 and 30
    '''
    images: batch of contact map (B,max_len,max_len)
    dis1,dis2,dis3 are the thresholds for contact maps for 3 channels
    '''
    images = np.expand_dims(images, axis=1)
    images = np.repeat(images, 3, axis=1)  # because the current REsnet work on 3 channels
    images[:, 0, :, :] = images[:, 0, :, :] * (images[:, 0, :, :] <= dis1)
    images[:, 1, :, :] = images[:, 1, :, :] * (images[:, 1, :, :] <= dis2)
    images[:, 2, :, :] = images[:, 2, :, :] * (images[:, 2, :, :] <= dis3)
    return images


# The second method to convert contact map
def contact_3Dchannel_v2(images, dis1=10.0, dis2=20.0, dis3=30.0):  # change in train and validation function
    '''
    images: batch of contact map (B,max_len,max_len)
    dis1,dis2,dis3 are the thresholds for contact maps for 3 channels
    '''
    images = np.expand_dims(images, axis=1)
    images = np.repeat(images, 3, axis=1)  # because the current REsnet work on 3 channels
    images[:, 0, :, :] = (dis1 - np.clip(images[:, 0, :, :], 0, dis1)) / dis1
    images[:, 1, :, :] = (dis2 - np.clip(images[:, 1, :, :], 0, dis2)) / dis2
    images[:, 2, :, :] = (dis3 - np.clip(images[:, 2, :, :], 0, dis3)) / dis3
    return images


# original contact map
def contact_3Dchannel_v3(images, dis1=22, dis2=22, dis3=22):  # change in train and validation function
    '''
    images: batch of contact map (B,max_len,max_len)
    dis1,dis2,dis3 are the thresholds for contact maps for 3 channels
    '''
    images = np.expand_dims(images, axis=1)
    images = np.repeat(images, 3, axis=1)  # because the current REsnet work on 3 channels
    images[:, 0, :, :] = (np.clip(images[:, 0, :, :], 0, dis1))  # /dis1 #normalize is not good
    images[:, 1, :, :] = (np.clip(images[:, 1, :, :], 0, dis2))  # /dis2
    images[:, 2, :, :] = (np.clip(images[:, 2, :, :], 0, dis3))  # /dis3
    return images


# original contact map normalize by z-score
def contact_3Dchannel_v6_zscore(images, dis1=22, dis2=22, dis3=22, std=1):  # change in train and validation function
    '''
    images: batch of contact map (B,max_len,max_len)
    dis1,dis2,dis3 are the thresholds for contact maps for 3 channels
    '''
    images = np.expand_dims(images, axis=1)
    images = np.repeat(images, 3, axis=1)  # because the current REsnet work on 3 channels
    images[:, 0, :, :] = (np.clip(images[:, 0, :, :], 0, dis1) - dis1) / std
    images[:, 1, :, :] = (np.clip(images[:, 1, :, :], 0, dis2) - dis2) / std  # /dis2
    images[:, 2, :, :] = (np.clip(images[:, 2, :, :], 0, dis3) - dis3) / std  # /dis3
    return images


def contactmap_loss(preds, labels, mu=None, logvar=None, n_nodes=None, norm=1, pos_weight=1):
    # cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight) #in my labels are distance not similarities
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
