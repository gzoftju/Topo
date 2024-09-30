import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data
from tqdm import tqdm
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
    feature_prop,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, LoraModel, PeftConfig, PeftModel, inject_adapter_in_model
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW, get_linear_schedule_with_warmup
from all_graph_templates import template_train,template_test,template_link
from torch.utils.data import DataLoader, Dataset, Sampler
import dgl
import os
import pickle
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
import json
import random
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
        
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  
    
def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')


    # CPU/GPU
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=3)
    parser.add_argument("--device", type=int, default=3, help="CUDA device, -1 means CPU")
    

    # Model Config
    parser.add_argument('--backbone', type=str, default='/data/guanzhong2/llama2-7b-hf')
    parser.add_argument('--tokenizer', type=str, default='LlamaTokenizer')
    parser.add_argument('--max_text_length', type=int, default=4096)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=int, default=0.05)

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=16)

    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--tau', type=int, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="5,10",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    # Inference
    parser.add_argument('--num_beams', type=int, default=20)
    #parser.add_argument('--gen_max_length', type=int, default=64)

    """Dataset"""
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    return args

######
#���ݼ�
######
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def get_loader(args,mode):
    tokenizer = LlamaTokenizer.from_pretrained(args.backbone, max_length=args.max_text_length,do_lower_case=args.do_lower_case)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    dataset = LLM_Dataset(args=args,tokenizer=tokenizer,mode=mode)

    if mode == 'train':
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            sampler=None,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            worker_init_fn=worker_init_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=args.num_workers, pin_memory=True,sampler=None,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False, worker_init_fn=worker_init_fn)

    return loader

class LLM_Dataset(Dataset):
    def __init__(self,args,tokenizer,mode):
        self.g, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,)
        self.tokenizer=tokenizer
        set_seed(args.seed)
        self.g.create_formats_()
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in args.fan_out.split(",")]
        )
        self.args=args
        self.mode=mode
        self.dict_nodeidx2title=load_pickle('./data/ogbn_arxiv/dict_nodeidx2title.pkl')
        self.dict_labelid2arxivcategeory=load_pickle('./data/ogbn_arxiv/dict_labelid2arxivcategeory.pkl')
        
        
        
        self.gumbel_node_all=np.load("./outputs/transductive/ogbn-arxiv/SAGE/seed_0/gumbel_emb_node.npz")['arr_0']
        self.gumbel_link_all=np.load("./outputs/transductive/ogbn-arxiv/SAGE/seed_0/gumbel_emb_link.npz")['arr_0']
        
        self.data_list=[]
        #with open("data_text.json", "r") as json_file:
        #      self.data_list = self.data_list+json.load(json_file)
        with open("data_main.json", "r") as json_file:.
              self.data_list = self.data_list+json.load(json_file)
        #with open("data_struct.json", "r") as json_file:
        #      self.data_list = self.data_list+json.load(json_file)
        #with open("data_sim.json", "r") as json_file:
        #      self.data_list = self.data_list+json.load(json_file) 
        self.data_list=self.data_list
        
        if self.mode=='train':
            self.dataloader = dgl.dataloading.DataLoader(
            self.g,
            self.idx_train,
            self.sampler,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,)
            
            self.total_length=len(self.data_list)
        else:
            self.dataloader = dgl.dataloading.DataLoader(
                self.g,
                self.idx_test,
                self.sampler,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
            )
            self.total_length=len(self.idx_test)
        self.dataloader_iter = iter(self.dataloader)
    def __len__(self):
        return self.total_length
    def __getitem__(self, idx):
        if random.random()>=0.5 and self.mode=='train':
            input_nodes, output_nodes, blocks = next(self.dataloader_iter)
            # input_nodes, output_nodes, blocks=self.dataloader[idx]
            blocks = [blk.int() for blk in blocks]
            src_ids_0, dst_ids_0 = blocks[0].all_edges()
            src_ids_1, dst_ids_1 = blocks[1].all_edges()
            src_0 = blocks[0].srcdata[dgl.NID][src_ids_0].tolist()
            out_0 = blocks[0].dstdata[dgl.NID][dst_ids_0].tolist()
            src_1 = blocks[1].srcdata[dgl.NID][src_ids_1].tolist()
            out_1 = blocks[1].dstdata[dgl.NID][dst_ids_1].tolist()
            src = np.array(src_0 + src_1)
            out = np.array(out_0 + out_1)
    
            first_order = np.concatenate((src[out == output_nodes.item()], out[src == output_nodes.item()]))
            first_order = np.unique(first_order[first_order != output_nodes.item()])
            
            random_index = np.random.randint(len(first_order))
            labels = first_order[random_index]
            first_order = np.delete(first_order, random_index)

            second_order = {}
            for i in first_order:
                temp = np.concatenate((src[out == i], out[src == i]))
                temp = np.unique(temp[(temp != i) & (temp != output_nodes.item())])
                second_order[i] = temp
    
            parse_matrix = ""
            for i in second_order.keys():
                parse_matrix += str(self.dict_nodeidx2title[i]) + ' is connected [' + ','.join(
                    str(self.dict_nodeidx2title[item]) for item in second_order[i]) + '],'
            parse_matrix = '[' + parse_matrix[:-1] + ']'
            
            
            
            labels = self.dict_nodeidx2title[labels.item()]
            
            gumbel_node = self.gumbel_node_all[output_nodes.item()]
            gumbel_link = self.gumbel_link_all[output_nodes.item()]
            
            output_nodes = self.dict_nodeidx2title[output_nodes.item()]
    
            if self.mode == 'train':
                input_text = template_link['source'].format(output_nodes, parse_matrix, output_nodes, labels)
                input_ids = self.tokenizer.encode(input_text, padding="longest", truncation=True,
                                                  max_length=self.args.max_text_length) + [2]
                
            else:
                input_text = template_link['source'].format(output_nodes, parse_matrix, output_nodes)
                input_ids = self.tokenizer.encode(input_text, padding="longest", truncation=True,
                                                  max_length=self.args.max_text_length)
    
            # print(self.tokenizer.tokenize(labels))
            labels_id = self.tokenizer.encode(labels)[1:] + [2]  # 去除开始加上结束
            target_ids = [-100] * (len(input_ids) - len(labels_id)) + labels_id
    
            out_dict = {}
            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
            out_dict['labels'] = labels
            out_dict['gumbel'] = gumbel_link
        else:
            input_nodes, output_nodes, blocks = next(self.dataloader_iter)
            # input_nodes, output_nodes, blocks=self.dataloader[idx]
            blocks = [blk.int() for blk in blocks]
            src_ids_0, dst_ids_0 = blocks[0].all_edges()
            src_ids_1, dst_ids_1 = blocks[1].all_edges()
            src_0 = blocks[0].srcdata[dgl.NID][src_ids_0].tolist()
            out_0 = blocks[0].dstdata[dgl.NID][dst_ids_0].tolist()
            src_1 = blocks[1].srcdata[dgl.NID][src_ids_1].tolist()
            out_1 = blocks[1].dstdata[dgl.NID][dst_ids_1].tolist()
            src = np.array(src_0 + src_1)
            out = np.array(out_0 + out_1)
    
            first_order = np.concatenate((src[out == output_nodes.item()], out[src == output_nodes.item()]))
            first_order = np.unique(first_order[first_order != output_nodes.item()])
    
            second_order = {}
            for i in first_order:
                temp = np.concatenate((src[out == i], out[src == i]))
                temp = np.unique(temp[(temp != i) & (temp != output_nodes.item())])
                second_order[i] = temp
    
            parse_matrix = ""
            for i in second_order.keys():
                parse_matrix += str(self.dict_nodeidx2title[i]) + ' is connected [' + ','.join(
                    str(self.dict_nodeidx2title[item]) for item in second_order[i]) + '],'
            parse_matrix = '[' + parse_matrix[:-1] + ']'
    
            labels = self.dict_labelid2arxivcategeory[self.labels[output_nodes].item()]
            
            print(output_nodes)
            gumbel_node = self.gumbel_node_all[output_nodes.item()]
            gumbel_link = self.gumbel_link_all[output_nodes.item()]
            
            output_nodes = self.dict_nodeidx2title[output_nodes.item()]
    
            if self.mode == 'train':
                input_text = template_train['source'].format(output_nodes, parse_matrix, output_nodes, labels)
                input_ids = self.tokenizer.encode(input_text, padding="longest", truncation=True,
                                                  max_length=self.args.max_text_length) + [2]
                
            else:
                input_text = template_test['source'].format(output_nodes, parse_matrix, output_nodes)
                input_ids = self.tokenizer.encode(input_text, padding="longest", truncation=True,
                                                  max_length=self.args.max_text_length)
    
            # print(self.tokenizer.tokenize(labels))
            labels_id = self.tokenizer.encode(labels)[1:] + [2]  # 去除开始加上结束
            target_ids = [-100] * (len(input_ids) - len(labels_id)) + labels_id
    
            out_dict = {}
            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
            out_dict['labels'] = labels
            out_dict['gumbel'] = gumbel_node
        return out_dict
    def collate_fn(self, batch):  # This funcion will be called after the '__getitem__' to organize the real batch data.
        batch_entry = {}
        B = len(batch)
        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * (-100)

        labels=[]
        gumbel=[]
        for i, entry in enumerate(batch):
            input_ids[i, -entry['input_length']:] = entry['input_ids']
            target_ids[i, -entry['target_length']:] = entry['target_ids']
            labels.append(entry['labels'])
            gumbel.append(entry['gumbel'])
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id).to(dtype=input_ids.dtype).cpu()  # attention mask
        batch_entry['attn_mask'] = attn_mask
        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['labels'] = labels
        batch_entry['gumbel'] = torch.tensor(gumbel)
        
        return batch_entry  # Real batch data.

####
#�Ż���
####
def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer
    

#######
#ѵ����
######
class Trainer(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None,train=True):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.verbose = True     # Only the main GPU's verbose == True
        model_class = LlamaForCausalLM
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class)
        self.de_model=nn.Linear(4096,4096)
        self.codebook=torch.tensor(np.load("./outputs/transductive/ogbn-arxiv/SAGE/seed_0/codebook_embeddings.npz")['arr_0'])
        self.model.tokenizer = self.tokenizer
        #self.accelerator = Accelerator()
        
        
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        #self.model = self.model.to(args.device)
        # Optimizer: AdamW
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
        #self.train_loader, self.val_loader, self.model, self.optim = self.accelerator.prepare(self.train_loader, self.val_loader, self.model, self.optim)
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def create_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU {self.args.gpu}')
        model_name = self.args.backbone
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            load_in_8bit=True,
            
            torch_dtype=torch.float16,
            **kwargs
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']
        )
        model= get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model
    def create_tokenizer(self, **kwargs):
        from transformers import LlamaTokenizer
        tokenizer_class = LlamaTokenizer
        tokenizer_name = self.args.backbone
        if True:
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case,
                **kwargs
                )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        if 'adamw' in self.args.optim:
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup

            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)
            
            if self.verbose:
                print("Batch per epoch: %d" % batch_per_epoch)
                print("Total Iters: %d" % t_total)
                print('Warmup ratio:', warmup_ratio)
                print("Warm up Iters: %d" % warmup_iters)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in self.de_model.named_parameters() if p.requires_grad],
                    "weight_decay": 0.0,
                }
            ]
            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
            lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)
        else:
            optim = self.args.optimizer(
                list(self.model.parameters()), self.args.lr)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path,de_path):
        results_1 = self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
        results_2 = self.de_model.load_state_dict(torch.load(de_path), strict=True)  
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            print(results_1)
            print('Model loaded from ', de_path)
            print(results_2)
    def predict(self):
        pass
    def evaluate(self):
        pass
    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))
    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)
        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)
    def train(self):
        # ckpt_path="llm_0_end.pth"
        # de_path="de_0_end.pth"
        # self.load_checkpoint(ckpt_path,de_path)
        print(self.args.epoch)
        for epoch in range(self.args.epoch):
            # Train
            self.model.train()
            self.de_model.train()
            self.de_model=self.de_model.cuda()
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)
            for step_i, batch in enumerate(self.train_loader):
                dddd = next(self.model.parameters()).device
                input_ids = batch['input_ids'].to(dddd)
                lm_labels = batch["target_ids"].to(dddd)
                attn_mask = batch["attn_mask"].to(dddd)
                B, L = lm_labels.size()
                # forward
                #print(input_ids.shape)
                #print(input_ids[:,:100])
                #print(attn_mask[:,:100])
                #print(lm_labels[:,-10:])
                #print(self.tokenizer.batch_decode(input_ids[:,-20:]))
                #print(input_ids[:,-20:])
                #print(lm_labels[:,-20:])
                
                output= self.model(
                    input_ids=input_ids,
                    labels=lm_labels,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True)
                
                last_layer_embedding=output.hidden_states[-1]
                last_layer_embedding=last_layer_embedding[:,-2,:]
                logits=self.de_model(last_layer_embedding)
                soft_one_hot=F.gumbel_softmax(logits, tau=1, dim=1, hard=False)
                
                z_q = einsum('b n, n d -> b d', soft_one_hot, self.codebook.cuda())
                
                sim_loss=0.5*F.mse_loss(z_q,batch['gumbel'].cuda())
                
                
                loss = output['loss']+sim_loss
                print(output['loss'])
                print(sim_loss)
                loss = loss / self.args.gradient_accumulation_steps
                #self.accelerator.backward(loss)
                loss.backward()
                loss=loss.detach()
                if step_i==4000:
                    torch.save(self.model.state_dict(),"llm_{}_mid1.pth".format(epoch))
                    torch.save(self.de_model.state_dict(),"de_{}_mid1.pth".format(epoch))     
                if step_i==8000:
                    torch.save(self.model.state_dict(),"llm_{}_mid2.pth".format(epoch)) 
                    torch.save(self.de_model.state_dict(),"de_{}_mid2.pth".format(epoch)) 
                if step_i==22000:
                    torch.save(self.model.state_dict(),"llm_{}_mid3.pth".format(epoch)) 
                    torch.save(self.de_model.state_dict(),"de_{}_mid3.pth".format(epoch))       
                if step_i % self.args.gradient_accumulation_steps == 0:

                    parameters=list(self.model.parameters())+list(self.de_model.parameters())
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    # torch.nn.utils.clip_grad_norm_(self.de_model.parameters(), self.args.clip_grad_norm)
                    torch.nn.utils.clip_grad_norm_(parameters, self.args.clip_grad_norm)


                    self.optim.step()  # Update
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    # # Zero out gradients to prepare for the next iteration
                    for param in self.model.parameters():
                        param.grad = None
                    for param in self.de_model.parameters():
                        param.grad = None
                if self.lr_scheduler:
                    lr = self.lr_scheduler.get_lr()[0]
                else:
                    lr=self.optim.param_groups[-1]['lr']
                if self.verbose and step_i % 1 == 0:  # Logging purpose
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'
                    desc_str += f' Loss:{loss:.3f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)
            if self.verbose:
                pbar.close()
            if self.verbose:  # Save checkpoint
                torch.save(self.model.state_dict(),"llm_{}_end.pth".format(epoch))
                torch.save(self.de_model.state_dict(),"de_{}_end.pth".format(epoch))

    def test(self):
        for epoch in range(3, 11, 4):
            if (epoch + 1) % 4 == 1:
                ckpt_path = "llm_{}_mid1.pth".format(epoch // 4)
                de_path = "de_{}_mid1.pth".format(epoch // 4)
            elif (epoch + 1) % 4 == 2:
                ckpt_path = "llm_{}_mid2.pth".format(epoch // 4)
                de_path = "de_{}_mid2.pth".format(epoch // 4)
            elif (epoch + 1) % 4 == 3:
                ckpt_path = "llm_{}_mid3.pth".format(epoch // 4)
                de_path = "de_{}_mid3.pth".format(epoch // 4)
            else:
                ckpt_path = "llm_{}_end.pth".format(epoch // 4)
                de_path = "de_{}_end.pth".format(epoch // 4)
            self.load_checkpoint(ckpt_path,de_path)
            #self.model = self.model.to(self.args.gpu)

            valid_results = self.evaluate_epoch()  # For metric
            msg = "NDCG={:.4f} | HR={:.4f} | Precision={:.4f} | Invalid users={}".format(
                np.mean(valid_results['avg_ndcg']), np.mean(valid_results['avg_hit']),
                np.mean(valid_results['avg_precision']), np.sum(valid_results['len(invalid_users)']))
            print(msg)
            if self.verbose:
                acc_file = open('llm_llama2-7b.txt', 'a')
                if (epoch + 1) % 4 == 1:
                    acc_file.write(str(epoch // 4) + '_mid1' + '\n')
                elif (epoch + 1) % 4 == 2:
                    acc_file.write(str(epoch // 4) + '_mid2' + '\n')
                elif (epoch + 1) % 4 == 3:
                    acc_file.write(str(epoch // 4) + '_mid3' + '\n')
                else:
                    acc_file.write(str(epoch // 4) + '_end' + '\n')
                acc_file.write(msg + '\n\n')
                acc_file.close()
    def evaluate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            results={'ACC':[]}
            acc=0
            print('len of val_loader is {}'.format(len(self.val_loader)))
            for step_i, batch in tqdm(enumerate(self.val_loader)):
                output= self.model.generate(input_ids=batch['input_ids'],attention_mask=batch['attn_mask'],max_new_tokens=20)

                output=self.tokenizer.batch_decode(output[:,-20:])
                print(output)
                for i in range(4):
                    if batch['labels'][i] in output[i]:
                       acc+=1
                print(acc)
                #print(self.tokenizer.batch_decode(batch['target_ids'][:,-3:]))
                #print(self.tokenizer.batch_decode(output[:,-20:]))
                #print(self.tokenizer.decode(batch['target_ids'][0]))
                #evaluation_result=evaluate_old(output,label)
                # 将结果累加到 result 的对应位置
                #results['ACC'].append(evaluation_result[0])
            return results

def main_worker(gpu, args):     # the gpu represents the local_rank
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')
    if False:
        print(f'Building train loader at GPU {gpu}')    # Train Consoles
        train_loader = get_loader(
            args,
            mode='train',
        )
        trainer = Trainer(args,train_loader= train_loader,  train=True)
        trainer.train()
    # define the prompts used in validation/ Test
    if True:
        val_loader = get_loader(
            args,
            mode='val',
        )
        trainer = Trainer(args, val_loader= val_loader, train=False)
        trainer.test()
def main():
    args = get_args()
    set_seed(args.seed)

    #当前可见的设备可使用如下代码来查看
    print(torch.cuda.device_count())
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"
    main_worker(device, args)
if __name__ == "__main__":
    main()