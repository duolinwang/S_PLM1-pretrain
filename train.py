import argparse
import torch
import os
import yaml
import time
from model import prepare_models, AverageMeter, reset_current_mixlen_batch, reset_current_samelen_batch
from model import check_samelen_batch, check_mixlen_batch, log_negative_mean_logtis
from model import MaskedLMDataCollator
from data import prepare_dataloaders
import numpy as np
from utils import prepare_optimizer, load_checkpoints, save_checkpoints, load_configs, test_gpu_cuda, prepare_saving_dir
from utils import get_logging, prepare_tensorboard
from utils import accuracy, pad_concatmap, Image_resize
from utils import contact_3Dchannel_v1, contact_3Dchannel_v2, contact_3Dchannel_v3, contact_3Dchannel_v6_zscore
from torch.cuda.amp import GradScaler, autocast

def training_loop(simclr,train_loader, val_loader, scaler, batch_converter, criterion,
                  optimizer_struct, optimizer_seq, scheduler_struct, scheduler_seq, writer,
                  result_path, logging, configs,masked_lm_data_collator=None):

    current_ratio = 0
    n_steps = 0
    n_sub_steps = 0
    epoch_num = 0
    train_alt=False
    if hasattr(configs.model.esm_encoder,"MLM"):
       if configs.model.esm_encoder.MLM.enable and configs.model.esm_encoder.MLM.alt:
          train_alt=True
    
    while True:
        epoch_num += 1
        #for epoch_num in range(configs.train_settings.epochs):
        logging.info(f"Epoch: {epoch_num}")
        
        losses = AverageMeter()
        if hasattr(configs.model.esm_encoder,"MLM"):
            if configs.model.esm_encoder.MLM.enable:
              MLM_losses = AverageMeter()
              simclr_losses = AverageMeter()
        
        bsz = configs.train_settings.batch_size
        batch_seq_mixlen, images_mixlen = [],[]#reset_current_mixlen_batch()
        batch_seq_samelen, images_samelen = [],[]#reset_current_samelen_batch()

        start = time.time()
        for batch_idx, batch_data in enumerate(iter(train_loader)):
            batch = batch_data[0]
            seqlen = len(str(batch[0]['seq']))
            if len(batch_seq_samelen) == 0:
                current_length = seqlen  # only set for the first sequence in batch
                # samelen_flag=True
            
            if np.abs(seqlen - current_length) <= configs.train_settings.length_range or configs.train_settings.same_diff_ratio==-1:
                images_samelen.append(batch[0]['contactmap'])
                batch_seq_samelen.append((batch[0]['index'], str(batch[0]['seq'])))
            else:
                # only save when mix data not beyond same_diff_ratio
                if current_ratio <= configs.train_settings.same_diff_ratio:
                    images_mixlen.append(batch[0]['contactmap'])
                    batch_seq_mixlen.append((batch[0]['index'], str(batch[0]['seq'])))

            if len(batch_seq_samelen)==bsz:
                samelen_flag = True
                batch_seq = batch_seq_samelen
                images = images_samelen
                current_ratio = 0  # one samelen batch, then reset current_ratio
            elif len(batch_seq_mixlen)==bsz:
                current_ratio += 1
                if current_ratio <= configs.train_settings.same_diff_ratio:
                    samelen_flag = False
                    batch_seq = batch_seq_mixlen
                    images = images_mixlen
                else:
                    continue
            else:
                continue
            
            print("n_steps="+str(n_steps))
            print("n_sub_steps="+str(n_sub_steps))
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            if configs.train_settings.rescale==1:
                pad_images = Image_resize(images, configs.model.esm_encoder.max_length)
            else:
                pad_images, mask_matrix = pad_concatmap(images, configs.model.esm_encoder.max_length)
                mask_matrix=torch.from_numpy(mask_matrix).float()
                mask_matrix=mask_matrix.to(configs.train_settings.device)
            
            if configs.train_settings.contactmethod.name == "contact_3Dchannel_v2":
                images = contact_3Dchannel_v2(pad_images, configs.train_settings.contactmethod.dist1, configs.train_settings.contactmethod.dist2, configs.train_settings.contactmethod.dist3)
            elif configs.train_settings.contactmethod.name == "contact_3Dchannel_v3":
                images = contact_3Dchannel_v3(pad_images, configs.train_settings.contactmethod.dist1, configs.train_settings.contactmethod.dist2, configs.train_settings.contactmethod.dist3)
            elif configs.train_settings.contactmethod.name == "contact_3Dchannel_v6_zscore":
                images = contact_3Dchannel_v6_zscore(pad_images, configs.train_settings.contactmethod.dist1, configs.train_settings.contactmethod.dist2, configs.train_settings.contactmethod.dist3)

            images = torch.tensor(images).float()
            images = images.to(configs.train_settings.device)  # copy the data to the device used for training
            batch_tokens = batch_tokens.to(configs.train_settings.device)
            with autocast(enabled=configs.train_settings.mixed_precision):
                simclr.model_seq.train()
                simclr.model_struct.train()
                loss=torch.tensor(0).float().to(configs.train_settings.device)
                if (train_alt and n_sub_steps%2==0) or not train_alt:
                    if configs.train_settings.rescale:
                       features_struct = simclr.model_struct(images)
                    else:
                       features_struct = simclr.model_struct(images,mask_matrix)
                    
                    features_seq,features_residue = simclr.model_seq(batch_tokens)
                    
                    if hasattr(configs.model, 'memory_banck'):
                        if configs.model.memory_banck.enable:
                            logits, labels=simclr.info_nce_loss_memory_bank(features_struct, features_seq)
                        else: 
                            logits, labels = simclr.info_nce_loss(features_struct, features_seq)
                    else:
                        logits, labels = simclr.info_nce_loss(features_struct, features_seq)
                    
                    simclr_loss = criterion(logits, labels)
                    loss += simclr_loss  # +contact_loss
                
                if (train_alt and n_sub_steps%2==1) or not train_alt:
                    if hasattr(configs.model.esm_encoder,"MLM"):
                      if configs.model.esm_encoder.MLM.enable:
                        # make masked input and label
                        mlm_inputs, mlm_labels = masked_lm_data_collator.mask_tokens(batch_tokens)
                        if hasattr(configs.model.esm_encoder.MLM,"mode") and configs.model.esm_encoder.MLM.mode=="contrast":
                            features_seq_mask,_ = simclr.model_seq(mlm_inputs)
                            if hasattr(configs.model, 'memory_banck'):
                                if configs.model.memory_banck.enable:
                                    logits_mask, labels_mask=simclr.info_nce_loss_memory_bank(features_seq_mask, features_seq)
                                else: 
                                    logits_mask, labels_mask = simclr.info_nce_loss(features_seq_mask, features_seq)
                            else:
                                logits_mask, labels_mask = simclr.info_nce_loss(features_seq_mask, features_seq)
                            
                            MLM_loss = criterion(logits_mask, labels_mask)
                        else:
                           prediction_scores = simclr.model_seq(mlm_inputs,return_logits=True)
                           # CrossEntropyLoss
                           vocab_size = simclr.model_seq.alphabet.all_toks.__len__()
                           MLM_loss = criterion(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1))
                        
                        loss += MLM_loss

                
                # Normalize loss for gradient accumulation
                loss = loss / configs.train_settings.gradient_accumulation

            # Accumulates scaled gradients.
            scaler.scale(loss).backward()
            if hasattr(configs.model.esm_encoder,"MLM"):
              if configs.model.esm_encoder.MLM.enable:
                    if train_alt:
                      if n_sub_steps%2==0:
                        simclr_losses.update(simclr_loss.item(),bsz)
                      if n_sub_steps%2==1:
                        MLM_losses.update(MLM_loss.item(),bsz)
                    else:
                      simclr_losses.update(simclr_loss.item(),bsz)
                      MLM_losses.update(MLM_loss.item(),bsz)


                  
            
            losses.update(loss.item(), bsz)
            n_sub_steps += 1

            # Gradient accumulation
            if n_sub_steps % configs.train_settings.gradient_accumulation == 0:
                # Perform the optimization step
                scaler.step(optimizer_struct)
                scaler.step(optimizer_seq)
                scaler.update()

                # Zero the parameter gradients
                optimizer_struct.zero_grad()
                optimizer_seq.zero_grad()
                
                n_steps += 1

                # Step the schedulers
                scheduler_struct.step()
                scheduler_seq.step()

            if n_steps % configs.checkpoints_every == 0: # and n_steps != 0:
                # save checkpoint
                save_checkpoints(optimizer_struct, optimizer_seq, result_path, simclr, n_steps, logging, epoch_num,configs.model.memory_banck.enable)

            if n_steps % configs.valid_settings.do_every == 0: # and n_steps != 0: I want to see the step evaluations 
                l_prob = logits[:, 0].mean()
                negsim_struct_struct = log_negative_mean_logtis(logits, "struct_struct", bsz)
                negsim_struct_seq = log_negative_mean_logtis(logits, "struct_seq", bsz)
                # negsim_seq_struct = log_negative_mean_logtis(logits,"seq_struct",bsz)
                negsim_seq_seq = log_negative_mean_logtis(logits, "seq_seq", bsz)

                top1, top5 = accuracy(logits, labels, topk=(1, 1))
                #logging.info(f'evaluation - step {n_steps}')
                loss_val, val_l_prob, val_negsim_struct_struct, val_negsim_struct_seq, val_negsim_seq_seq = evaluation_loop(simclr, val_loader, batch_converter, criterion, configs,masked_lm_data_collator=masked_lm_data_collator)
                #logging.info(f"\tSame_len: {samelen_flag}, Current_len: {current_length:.0f}, Loss: {loss:.4f}, Top1 accuracy: {top1[0]}, l_prob: {l_prob.item():.2f}, N_stru_stru: {negsim_struct_struct:.2f}, N_stru_seq: {negsim_struct_seq:.2f}, N_seq_seq: {negsim_seq_seq:.2f}, "
                #             f"Validation: {loss_val:.4f}, val_l_prob: {val_l_prob:.2f}, val_N_stru_stru: {val_negsim_struct_struct:.2f}, val_N_stru_seq: {val_negsim_struct_seq:.2f}, val_N_seq_seq: {val_negsim_seq_seq:.2f}")
                logging.info(f"step:{n_steps} Same_len:{samelen_flag}, Current_len: {current_length:.0f}, Loss: {loss:.4f}, Top1 accuracy: {top1[0]}, l_prob: {l_prob.item():.2f}, N_stru_stru: {negsim_struct_struct:.2f}, N_stru_seq: {negsim_struct_seq:.2f}, N_seq_seq: {negsim_seq_seq:.2f}, "
                             f"Validation: {loss_val:.4f}, val_l_prob: {val_l_prob:.2f}, val_N_stru_stru: {val_negsim_struct_struct:.2f}, val_N_stru_seq: {val_negsim_struct_seq:.2f}, val_N_seq_seq: {val_negsim_seq_seq:.2f}")
                
                logging.info(f"step:{n_steps}\tTrain_loss_avg: {losses.avg:.6f}, lr: {scheduler_struct.get_lr()[0]:.2e}")
                if hasattr(configs.model.esm_encoder,"MLM"):
                    if configs.model.esm_encoder.MLM.enable:
                            if hasattr(configs.model.esm_encoder.MLM,"mode") and configs.model.esm_encoder.MLM.mode=="contrast":
                               logging.info(f"step:{n_steps} Train_simclr_loss_avg: {simclr_losses.avg:.6f},Train_MLM_loss_avg: {MLM_losses.avg:.6f}")
                            else:
                               top1_mlm, _ = accuracy(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1), topk=(1, 1))
                               #acc should be top1_mlm/mask_ratio
                               logging.info(f"step:{n_steps} Train_simclr_loss_avg: {simclr_losses.avg:.6f},Train_MLM_loss_avg: {MLM_losses.avg:.6f}, Train_MLM_acc: {top1_mlm}")
                
                
                writer.add_scalar('loss', loss, global_step=n_steps)
                writer.add_scalar('acc/top1', top1[0], global_step=n_steps)
                writer.add_scalar('learning_rate_struct', scheduler_struct.get_lr()[0], global_step=n_steps)
                writer.add_scalar('learning_rate_seq', scheduler_seq.get_lr()[0], global_step=n_steps)
            
            if len(batch_seq_samelen) == bsz:
                #reset_current_samelen_batch()
                batch_seq_samelen,images_samelen=[],[]
            else:
                #reset_current_mixlen_batch()
                batch_seq_mixlen,images_mixlen=[],[]

            if n_steps > configs.train_settings.num_steps:
                break

        end = time.time()

        if n_steps > configs.train_settings.num_steps:
            break

        logging.info(f"one epoch cost {(end - start):.2f}, number of trained steps {n_steps}")


def evaluation_loop(simclr, val_loader, batch_converter, criterion, configs,masked_lm_data_collator=None):
    loss_val_sum = 0
    l_prob = 0
    negsim_struct_seq = 0
    negsim_seq_seq = 0
    negsim_struct_struct = 0
    k = 0
    for batch_data in iter(val_loader):
        batch = batch_data[0]
        bsz = len(batch)
        images = [d['contactmap'] for d in batch]
        batch_seq = [(d['index'], str(d['seq'])) for d in batch]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
        if configs.train_settings.rescale==1:
            pad_images = Image_resize(images, configs.model.esm_encoder.max_length)
        else:
            pad_images, mask_matrix = pad_concatmap(images, configs.model.esm_encoder.max_length)
            mask_matrix=torch.from_numpy(mask_matrix).float()
            mask_matrix=mask_matrix.to(configs.train_settings.device)
        
        if configs.train_settings.contactmethod.name == "contact_3Dchannel_v2":
            images = contact_3Dchannel_v2(pad_images, configs.train_settings.contactmethod.dist1, configs.train_settings.contactmethod.dist2, configs.train_settings.contactmethod.dist3)
        elif configs.train_settings.contactmethod.name == "contact_3Dchannel_v3":
            images = contact_3Dchannel_v3(pad_images, configs.train_settings.contactmethod.dist1, configs.train_settings.contactmethod.dist2, configs.train_settings.contactmethod.dist3)
        elif configs.train_settings.contactmethod.name == "contact_3Dchannel_v6_zscore":
            images = contact_3Dchannel_v6_zscore(pad_images, configs.train_settings.contactmethod.dist1, configs.train_settings.contactmethod.dist2, configs.train_settings.contactmethod.dist3)

        images = torch.tensor(images).float()
        images = images.to(configs.train_settings.device)  # copy the data to the device used for validation
        batch_tokens = batch_tokens.to(configs.train_settings.device)
        simclr.model_seq.eval()
        simclr.model_struct.eval()
        with torch.no_grad():
            if configs.train_settings.rescale:
               features_struct = simclr.model_struct(images)
            else:
               features_struct = simclr.model_struct(images,mask_matrix)
            
            features_seq,features_residue = simclr.model_seq(batch_tokens)
            logits, labels = simclr.info_nce_loss(features_struct, features_seq)
            loss_val_sum += criterion(logits, labels)  # +contact_loss/bsz
            l_prob += logits[:, 0].mean().item()
            negsim_struct_struct += log_negative_mean_logtis(logits, "struct_struct", bsz)
            negsim_struct_seq += log_negative_mean_logtis(logits, "struct_seq", bsz)
            negsim_seq_seq += log_negative_mean_logtis(logits, "seq_seq", bsz)
            if hasattr(configs.model.esm_encoder,"MLM"):
              if configs.model.esm_encoder.MLM.enable:
                # make masked input and label
                mlm_inputs, mlm_labels = masked_lm_data_collator.mask_tokens(batch_tokens)
                if hasattr(configs.model.esm_encoder.MLM,"mode") and configs.model.esm_encoder.MLM.mode=="contrast":
                    features_seq_mask,_ = simclr.model_seq(mlm_inputs)
                    logits_mask, labels_mask = simclr.info_nce_loss(features_seq_mask, features_seq)
                    masked_lm_loss = criterion(logits_mask, labels_mask)
                else:
                    prediction_scores = simclr.model_seq(mlm_inputs,return_logits=True)
                    # CrossEntropyLoss
                    vocab_size = simclr.model_seq.alphabet.all_toks.__len__()
                    masked_lm_loss = criterion(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1))
                
                loss_val_sum += masked_lm_loss
            
            k = k + 1

    loss_val_sum = loss_val_sum / float(k)
    l_prob = l_prob / float(k)
    negsim_struct_struct = negsim_struct_struct / float(k)
    negsim_struct_seq = negsim_struct_seq / float(k)
    negsim_seq_seq = negsim_seq_seq / float(k)

    return float(loss_val_sum), float(l_prob), float(negsim_struct_struct), float(negsim_struct_seq), float(
        negsim_seq_seq)


def main(args, dict_configs,config_file_path):
    print("start!")

    configs = load_configs(dict_configs,args)
    if args.seed:
        configs.fix_seed = args.seed
    
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    test_gpu_cuda()

    result_path, checkpoint_path = prepare_saving_dir(configs,config_file_path)

    logging = get_logging(result_path)

    train_writer = prepare_tensorboard(result_path)

    train_loader, val_loader = prepare_dataloaders(logging, configs)


    #model_seq, model_struct = prepare_models(logging, configs)
    simclr = prepare_models(logging, configs)

    scheduler_seq, scheduler_struct, optimizer_seq, optimizer_struct = prepare_optimizer(
        simclr.model_seq, simclr.model_struct, logging, configs
    )

    alphabet = simclr.model_seq.alphabet
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=configs.model.esm_encoder.max_length)
    if hasattr(configs.model.esm_encoder,"MLM") and configs.model.esm_encoder.MLM.enable:
       masked_lm_data_collator = MaskedLMDataCollator(batch_converter, mlm_probability=configs.model.esm_encoder.MLM.mask_ratio)
    else:
       masked_lm_data_collator=None
    
    simclr = load_checkpoints(
        simclr, configs,logging
    )

    scaler = GradScaler(enabled=configs.train_settings.mixed_precision)
    criterion = torch.nn.CrossEntropyLoss().to(configs.train_settings.device)

    logging.info(f"Start Contrastive training for {configs.train_settings.num_steps} steps.")
    logging.info(f"Training with: {configs.train_settings.device} and fix_seed = {configs.fix_seed}")
    
    training_loop(
        simclr, train_loader, val_loader, scaler, batch_converter, criterion,
        optimizer_struct, optimizer_seq, scheduler_struct, scheduler_seq, train_writer,
        result_path, logging, configs,masked_lm_data_collator=masked_lm_data_collator
    )

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    parser.add_argument("--result_path",default=None,help="result_path, if setted by command line, overwrite the one in config.yaml, by default is None")
    parser.add_argument("--resume_path",default=None,help="if set, overwrite the one in config.yaml, by default is None")
    parser.add_argument("--num_end_adapter_layers",default=None,help="num_end_adapter_layers")
    parser.add_argument("--module_type",default=None,help="module_type for adapterh")
    parser.add_argument("--seed",default=None,type=int,help="random seed")


    args_main = parser.parse_args()
    config_path = args_main.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(args_main, config_file,config_path)
