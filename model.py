import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time
from transformers import Swinv2Config, Swinv2Model
import esm
import esm_adapterH
from peft import PeftModel, LoraConfig, get_peft_model

import math

def get_embedding(embed_dim,num_embeddings):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        
        return emb

def patch_num_32(n):
        remainder = n % 32
        if remainder == 0:
            return n//32
        else:
            return (n + (32 - remainder))//32


def save_lora_checkpoint(model, save_path):
    """
    Save lora weights as a checkpoint in the save_path directory.
    """
    model.save_pretrained(save_path)


def load_and_add_lora_checkpoint(base_model, lora_checkpoint_path):
    """Add a pretrained LoRa checkpoint to a base model"""
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    return lora_model


def merge_lora_with_base_model(lora_model):
    """
    This method merges the LoRa layers into the base model.
    This is needed if someone wants to use the base model as a standalone model.
    """
    model = lora_model.merge_and_unload()
    return model


def remove_lora(lora_model):
    """
    Gets back the base model by removing all the lora modules without merging
    """
    base_model = lora_model.unload()
    return base_model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_negative_mean_logtis(logits, mode, minibatch_size):
    N = minibatch_size
    if mode == "struct_struct":
        a11 = logits[0:N, 1:N]
        return a11.mean().item()

    if mode == "struct_seq":
        a12 = logits[0:N, N:2 * N - 1]
        return a12.mean().item()

    if mode == "seq_struct":
        a21 = logits[N:2 * N, 1:N]
        return a21.mean().item()
    if mode == "seq_seq":
        a22 = logits[N:2 * N, N:2 * N - 1]
        return a22.mean().item()


def get_pretrainweights_swinv2(swinpretrain_name, swinpretrain_local=None):
    if swinpretrain_local is not None:
        backbone = Swinv2Model.from_pretrained(swinpretrain_name, cache_dir=swinpretrain_local)
    else:
        backbone = Swinv2Model.from_pretrained(swinpretrain_name)

    return backbone


class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x


class Swinv2(nn.Module):  # embedding table can be tuned
    def __init__(self, init_model, configs, logging, inner_dim=4096, out_dim=256, num_projector=2, seqlen=512,
                 unfix_last_layer=2):
        super(Swinv2, self).__init__()
        self.config = Swinv2Config(image_size=seqlen, patch_size=init_model.config.patch_size,window_size=init_model.config.window_size,hidden_size=init_model.config.hidden_size,
                    depths=init_model.config.depths,num_heads=init_model.config.num_heads,embed_dim=init_model.config.embed_dim)
        
        self.backbone = Swinv2Model(self.config)
        pretrained_state_dict = init_model.state_dict()
        self.backbone.load_state_dict(pretrained_state_dict,strict=False)
        
        dim_mlp = self.backbone.config.hidden_size
        # backbone_projector = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
        self.esm_pool_mode= configs.model.esm_encoder.pool_mode
        self.backbone_projector = MoBYMLP(in_dim=dim_mlp, inner_dim=inner_dim, out_dim=out_dim,
                                          num_layers=num_projector)
        if self.esm_pool_mode ==3:
           self.pe = get_embedding(dim_mlp,patch_num_32(seqlen)**2).to(configs.train_settings.device)
        
        for p in self.backbone.parameters():
            p.requires_grad = False  # fix all previous layers

        '''
        for name, module in model.named_modules():
            print(f"Module Name: {name}")
            print(module)
            print("=" * 50)
        '''
        self.num_layers = self.backbone.num_layers
        if configs.model.swin_encoder.lora.enable:
            logging.info('use lora for swin')
            if configs.model.swin_encoder.lora.resume.enable:
                self.swin2 = load_and_add_lora_checkpoint(
                    self.backbone, configs.model.swin_encoder.lora.resume.checkpoint
                )
            else:
                target_modules = ["attention.self.query", "attention.self.key", "attention.self.value",
                                  "attention.output.dense"]
                peft_config = LoraConfig(
                    inference_mode=False,
                    r=configs.model.swin_encoder.lora.r,
                    lora_alpha=configs.model.swin_encoder.lora.alpha,
                    target_modules=target_modules,
                    lora_dropout=configs.model.swin_encoder.lora.dropout,
                    bias="none",
                    # modules_to_save=modules_to_save
                )

                self.peft_model = get_peft_model(self.backbone, peft_config)

        else:
            logging.info('fine-tune swin v2 model')
            fix_layer_num = np.max([self.num_layers - unfix_last_layer, 0])
            fix_layer_index = 0
            for layer in self.backbone.encoder.layers:  # only fine-tune transformer layers
                if fix_layer_index < fix_layer_num:
                    fix_layer_index += 1
                    continue

                for p in layer.parameters():
                    # print("unfix")
                    p.requires_grad = True

            if unfix_last_layer != 0:  # if need fine-tune last layer, the layernorm for last representation should updated
                for p in self.backbone.layernorm.parameters():
                    # print("unfix layernorm")
                    p.requires_grad = True

        if configs.model.swin_encoder.tune_swin_table:
            for p in self.backbone.embeddings.parameters():  # the embedding table can be fine-tuned
                # print("unfix embedding table")
                p.requires_grad = True

    def forward(self, image,mask=None):
        if mask is None:
           outputs = self.backbone(image).pooler_output
        else:
           outputs = self.backbone(image).last_hidden_state
           pooled_mask =  nn.functional.max_pool2d(mask.unsqueeze(1), kernel_size=32, stride=32,ceil_mode=True).squeeze(1).view(mask.shape[0], -1)
           denom = torch.sum(pooled_mask,-1,keepdim=True)
           if self.esm_pool_mode!=3:
              outputs=torch.sum(outputs * pooled_mask.unsqueeze(-1), dim=1) / denom 
           elif self.esm_pool_mode==3:
                #pe =  get_embedding(outputs.shape[-1],outputs.shape[-2])
                outputs = torch.sum(outputs * self.pe.unsqueeze(0) * pooled_mask.unsqueeze(-1), dim=1)/denom
            
        embeddings = self.backbone_projector(outputs)
        return embeddings


class ESM2(nn.Module):  # embedding table is fixed
    def __init__(self, esm2_pretrain, logging,
                 # esm2_pretrain_local,
                 configs, inner_dim=4096, out_dim=256, num_projector=2,
                 unfix_last_layer=4):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(ESM2, self).__init__()
        if configs.model.esm_encoder.adapter_h.enable:
            logging.info("use adapter H")
            #num_end_adapter_layers = configs.model.esm_encoder.adapter_h.num_end_adapter_layers
            adapter_args = configs.model.esm_encoder.adapter_h
            esm2_dict = {"esm2_t33_650M_UR50D": esm_adapterH.pretrained.esm2_t33_650M_UR50D(adapter_args),  # 33 layers embedding=1280
                         "esm2_t30_150M_UR50D": esm_adapterH.pretrained.esm2_t30_150M_UR50D(adapter_args),  # 30 layers embedding=640
                         "esm2_t12_35M_UR50D": esm_adapterH.pretrained.esm2_t12_35M_UR50D(adapter_args),  # 12 layers embedding=480
                         "esm2_t6_8M_UR50D": esm_adapterH.pretrained.esm2_t6_8M_UR50D(adapter_args),  # 6 layers embedding = 320
                         }
        else:
            esm2_dict = {"esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D(),  # 33 layers embedding=1280
                         "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D(),  # 30 layers embedding=640
                         "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D(),  # 12 layers embedding=480
                         "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D(),  # 6 layers embedding = 320
                         }

        # if esm2_pretrain_local is None:
        self.esm2, self.alphabet = esm2_dict[esm2_pretrain]  # alphabet.all_toks
        # else:
        #     print("load esm2 model from local dir")
        #     self.esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_pretrain_local)

        self.num_layers = self.esm2.num_layers
        for p in self.esm2.parameters():  # frozen all parameters first
            p.requires_grad = False
        
        if configs.model.esm_encoder.adapter_h.enable:
            for name, param in self.esm2.named_parameters():
                if "adapter_layer" in name:
                  print("unfix adapter_layer")
                  param.requires_grad = True
        
        if configs.model.esm_encoder.lora.enable:
            logging.info('use lora for esm v2')
            if configs.model.esm_encoder.lora.resume.enable:
                self.esm2 = load_and_add_lora_checkpoint(self.esm2, configs.model.esm_encoder.lora.resume)
            else:
                # if args.esm_num_end_lora > 0:
                #     target_modules = []
                #     start_layer_idx = np.max([self.num_layers - args.esm_num_end_lora, 0])
                #     for idx in range(start_layer_idx, self.num_layers):
                #         for layer_name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                #                            "self_attn.out_proj"]:
                #             target_modules.append(f"layers.{idx}.{layer_name}")
                lora_targets =  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj","self_attn.out_proj"]
                target_modules=[]
                if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                    start_layer_idx = np.max([self.num_layers - configs.model.esm_encoder.lora.esm_num_end_lora, 0])
                    for idx in range(start_layer_idx, self.num_layers):
                        for layer_name in lora_targets:
                            target_modules.append(f"layers.{idx}.{layer_name}")
                    
                #target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]
                peft_config = LoraConfig(
                    inference_mode=False,
                    r=configs.model.esm_encoder.lora.r,
                    lora_alpha=configs.model.esm_encoder.lora.alpha,
                    target_modules=target_modules,
                    lora_dropout=configs.model.esm_encoder.lora.dropout,
                    bias="none",
                    # modules_to_save=modules_to_save
                )
                self.peft_model = get_peft_model(self.esm2, peft_config)
        elif configs.model.esm_encoder.fine_tuning.enable:
            logging.info('fine-tune esm v2')
            unfix_last_layer = configs.model.esm_encoder.fine_tuning.unfix_last_layer
            fix_layer_num = self.num_layers - unfix_last_layer
            fix_layer_index = 0
            for layer in self.esm2.layers:  # only fine-tune transformer layers,no contact_head and other parameters
                if fix_layer_index < fix_layer_num:
                    fix_layer_index += 1  # keep these layers frozen
                    continue

                for p in layer.parameters():
                    #logging.info('unfix layer')
                    p.requires_grad = True

            if unfix_last_layer != 0:  # if need fine-tune last layer, the emb_layer_norm_after for last representation should updated
                for p in self.esm2.emb_layer_norm_after.parameters():
                    p.requires_grad = True
        
        if configs.model.esm_encoder.tune_ESM_table:
            logging.info("fine-tune esm embedding parameters")
            for p in self.esm2.embed_tokens.parameters():
                p.requires_grad = True
        
        if hasattr(configs.model.esm_encoder,"MLM"):
           if configs.model.esm_encoder.MLM.enable and configs.model.esm_encoder.MLM.mode=="predict":
              for p in self.esm2.lm_head.parameters():
                  p.requires_grad = True
        
        self.pool_mode = configs.model.esm_encoder.pool_mode
        if self.pool_mode ==3:
                self.pe = get_embedding(self.esm2.embed_dim,configs.model.esm_encoder.max_length+2).to(configs.train_settings.device)
        
        self.projectors = MoBYMLP(in_dim=self.esm2.embed_dim, inner_dim=inner_dim, num_layers=num_projector,
                                  out_dim=out_dim)

    def forward(self, x,return_logits=False):
        outputs = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)
        if return_logits:
           prediction_scores = outputs["logits"]
           return prediction_scores
        else:
            residue_feature = outputs['representations'][self.num_layers]
            if self.pool_mode==1: #CLS token 
                graph_feature = residue_feature[:, 0, :]
            elif self.pool_mode==2: #average pooling but remove padding tokens
               mask = (x != self.alphabet.padding_idx)  #use this in v2 training
               denom = torch.sum(mask, -1, keepdim=True)
               graph_feature = torch.sum(residue_feature * mask.unsqueeze(-1), dim=1) / denom #remove padding
            elif self.pool_mode==3:
                mask = (x != self.alphabet.padding_idx)  #use this in v2 training
                denom = torch.sum(mask, -1, keepdim=True)
                graph_feature = torch.sum(residue_feature* self.pe[:residue_feature.shape[1],:].unsqueeze(0) * mask.unsqueeze(-1), dim=1) / denom #remove padding and normlize
            
            graph_feature = self.projectors(graph_feature)
            return graph_feature, residue_feature


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) + 1) / 2
        return adj

"""
class CustomModel(nn.Module):
    def __init__(self, model_seq, model_struct):
        super(CustomModel, self).__init__()
        self.model_seq = model_seq
        self.model_struct = model_struct

    def forward(self, images, batch_tokens):
        features_struct = self.model_struct(images)
        features_seq, features_residue = self.model_seq(batch_tokens)  # notice the change due to the usage of esm
        return features_seq, features_residue, features_struct
"""

def reset_current_samelen_batch():
    batch_seq_samelen = []
    images_samelen = []
    return batch_seq_samelen, images_samelen


def check_samelen_batch(batch_size, batch_seq_samelen):
    if len(batch_seq_samelen) == batch_size:
        return True
    else:
        return False


def reset_current_mixlen_batch():
    batch_seq_mixlen = []
    images_mixlen = []
    return batch_seq_mixlen, images_mixlen


def check_mixlen_batch(batch_size, batch_seq_mixlen):
    if len(batch_seq_mixlen) == batch_size:
        return True
    else:
        return False


class MaskedLMDataCollator:
    """Data collator for masked language modeling.
    
    The idea is based on the implementation of DataCollatorForLanguageModeling at
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L751C7-L782
    """
    
    def __init__(self, batch_converter, mlm_probability=0.15):
        """_summary_

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.15.
        """
        self.mlm_probability = mlm_probability
        self.special_token_indices = [batch_converter.alphabet.cls_idx, 
                                batch_converter.alphabet.padding_idx, 
                                batch_converter.alphabet.eos_idx,  
                                batch_converter.alphabet.unk_idx, 
                                batch_converter.alphabet.mask_idx]
        self.vocab_size = batch_converter.alphabet.all_toks.__len__()
        self.mask_idx = batch_converter.alphabet.mask_idx
        

        
    def get_special_tokens_mask(self, tokens):
        return [1 if token in self.special_token_indices else 0 for token in tokens]
    
    def mask_tokens(self, batch_tokens):
        """make a masked input and label from batch_tokens.

        Args:
            batch_tokens (tensor): tensor of batch tokens
            batch_converter (tensor): batch converter from ESM-2.

        Returns:
            inputs: inputs with masked
            labels: labels for masked tokens
        """
        ## mask tokens
        inputs = batch_tokens.clone().to(batch_tokens.device)
        labels = batch_tokens.clone().to(batch_tokens.device)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [self.get_special_tokens_mask(val) for val in labels ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx
        
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size,
                                     labels.shape, dtype=torch.long).to(batch_tokens.device)
        
        #print(indices_random)
        #print(random_words)
        #print(inputs)
        
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



class SimCLR_model(nn.Module):
    def __init__(self, model_seq, model_struct,configs,logging):
        super(SimCLR_model, self).__init__()
        self.model_seq = model_seq
        self.model_struct = model_struct
        self.batch_size=configs.train_settings.batch_size
        self.temperature=configs.train_settings.temperature
        self.n_views = configs.train_settings.n_views
        self.device=configs.train_settings.device
        if hasattr(configs.model, 'memory_banck'):
          if configs.model.memory_banck.enable:
             logging.info(f"using memory_banck")
             self.K = int(configs.model.memory_banck.K*configs.train_settings.batch_size)
             out_dim = configs.model.out_dim
             self.register_buffer("seq_queue", torch.randn(out_dim, self.K))
             self.seq_queue = F.normalize(self.seq_queue, dim=0)
             self.register_buffer("seq_queue_ptr", torch.zeros(1, dtype=torch.long))
             self.register_buffer("struct_queue", torch.randn(out_dim, self.K))
             self.struct_queue = F.normalize(self.struct_queue, dim=0)
             self.register_buffer("struct_queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _dequeue_and_enqueue(self, keys,queue,queue_ptr):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)                     #original but mask without multiple GPU
        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        #print(self.K)
        #print(batch_size)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        #queue[:, ptr:ptr + batch_size] = keys.T
        # Create a detached view of the tensor
        detached_queue = queue.detach()
        # Perform the in-place operation on the detached tensor
        detached_queue[:, ptr:ptr + batch_size] = keys.T
        # Copy the modified detached tensor back to the original tensor
        queue = detached_queue
        #queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        queue_ptr[0] = ptr
    
    def info_nce_loss_memory_bank(self,features_struct,features_seq,update=True):
        features_struct = F.normalize(features_struct, dim=1)
        features_seq = F.normalize(features_seq, dim=1)
        #print(features_struct.shape)
        #print(features_seq.shape)
        l_pos = torch.einsum('nc,nc->n', [features_struct, features_seq]).unsqueeze(-1)
        l_neg_struct_struct = torch.einsum('nc,ck->nk', [features_struct, self.struct_queue.clone().detach().to(self.device)])
        l_neg_struct_seq = torch.einsum('nc,ck->nk', [features_struct, self.seq_queue.clone().detach().to(self.device)])
        l_neg_seq_struct = torch.einsum('nc,ck->nk', [features_seq, self.struct_queue.clone().detach().to(self.device)])
        l_neg_seq_seq = torch.einsum('nc,ck->nk', [features_seq, self.seq_queue.clone().detach().to(self.device)])
        # logits: Nx(1+4K)
        #logits = torch.cat([l_pos, l_neg_struct_struct,l_neg_struct_seq,l_neg_seq_struct,l_neg_seq_seq], dim=1)
        #for debug compare with simCLR loss
        logits1 = torch.cat([l_pos, l_neg_struct_struct,l_neg_struct_seq],dim=1)
        logits2=torch.cat([l_pos,l_neg_seq_struct,l_neg_seq_seq], dim=1)
        logits = torch.cat([logits1,logits2],dim=0)
        # apply temperature
        logits /= self.temperature
        #print(logits.shape)
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        if update: #don't update memory for validation set
           # dequeue and enqueue
           self._dequeue_and_enqueue(features_seq,self.seq_queue,self.seq_queue_ptr)
           self._dequeue_and_enqueue(features_struct,self.struct_queue,self.struct_queue_ptr)
        
        return logits, labels
    
    
    def info_nce_loss(self,features_struct, features_seq):  # loss function
    
        # print("batch_size="+str(self.args.batch_size))
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)],
                           dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        features_struct = F.normalize(features_struct, dim=1)
        features_seq = F.normalize(features_seq, dim=1)
        features = torch.cat([features_struct, features_seq], dim=0)
    
        similarity_matrix = torch.matmul(features, features.T)
    
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    
        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
    
        logits = logits / self.temperature
        return logits, labels
    

def print_trainable_parameters(model, logging):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_models(logging, configs):
    # Use ResNetSImCLR for contact map
    # Use ViT for contact map
    model_struct_init = get_pretrainweights_swinv2(
        configs.model.swin_encoder.model_name
    )
    model_struct = Swinv2(model_struct_init, inner_dim=configs.model.swin_encoder.inner_dim,
                          out_dim=configs.model.out_dim,
                          num_projector=configs.model.num_projector, seqlen=configs.model.esm_encoder.max_length,
                          unfix_last_layer=configs.model.swin_encoder.unfixswin_last_layer,
                          configs=configs, logging=logging)
    model_struct = model_struct.to(configs.train_settings.device)

    # Use ESM2 for sequence
    model_seq = ESM2(configs.model.esm_encoder.model_name,
                     # configs.esm2_pretrain_local,
                     inner_dim=configs.model.esm_encoder.inner_dim,
                     out_dim=configs.model.out_dim,
                     num_projector=configs.model.num_projector,
                     configs=configs, logging=logging)
    model_seq = model_seq.to(configs.train_settings.device)

    print_trainable_parameters(model_seq, logging)
    print_trainable_parameters(model_struct, logging)
    
    ##new_model = CustomModel(model_seq, model_struct)
    simclr=SimCLR_model(model_seq,model_struct,configs=configs,logging=logging)
    return simclr #model_seq, model_struct #, new_model


if __name__ == '__main__':
    print('test')
