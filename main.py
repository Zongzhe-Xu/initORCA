import os
import json
import argparse
import operator
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from timeit import default_timer
from attrdict import AttrDict
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
from otdd.pytorch.distance import DatasetDistance, FeatureCost


from task_configs import get_data, get_config, get_metric, get_optimizer, get_scheduler
from utils import count_params, count_trainable_params, calculate_stats, conv_init, embedder_init, embedder_placeholder, get_params_to_update
import copy


def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    if feats.shape[0] == 1: 
        feats = torch.cat([feats] * 10, 0)
    if len(feats.shape) > 2:
        feats = feats.mean(1)
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).long().to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)
    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    d = dist.distance(maxsamples = len(src_train_dataset)) * 1e-2
    return d


class wrapper2D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='base', train_epoch=0):
        super().__init__()
        self.classification = output_shape[1] != 1
        self.output_raw = True

        if weight == 'tiny':
            arch_name = "microsoft/swin-tiny-patch4-window7-224"
            embed_dim = 96
            output_dim = 768
            img_size = 224
        elif weight == 'base':
            arch_name = "microsoft/swin-base-patch4-window7-224-in22k"
            embed_dim = 128
            output_dim = 1024
            img_size = 224

        if self.classification:
            self.model = SwinForImageClassification.from_pretrained(arch_name)
            self.model.config.image_size = img_size
            self.model = SwinForImageClassification.from_pretrained(arch_name, config=self.model.config)
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity()
            self.classifier = nn.Linear(in_features=output_dim, out_features=output_shape[1])
        else:
            self.model = SwinForMaskedImageModeling.from_pretrained(arch_name)
            self.model.config.image_size = img_size
            self.model = SwinForMaskedImageModeling.from_pretrained(arch_name, config=self.model.config)
            self.pool = nn.AdaptiveAvgPool2d(input_shape[-2:])

        set_grad_state(self.model, False)

        if use_embedder:
            self.embedder = Embeddings2D(input_shape, output_shape,config=self.model.config, embed_dim=embed_dim, img_size=img_size)
            embedder_init(self.model.swin.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder  


    def forward(self, x):
        if self.output_raw:
            return self.model.swin.embeddings(x)[0]
            
        x = self.model(x).logits

        if self.classification:
            return self.classifier(x)
        else:
            b, c, h, w = x.shape
            return self.pool(x.mean(1))


class wrapper1D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='roberta', train_epoch=0):
        super().__init__()

        self.dense = False
        self.output_raw = True
        self.weight = weight

        if weight =='swin':
            self.model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.model.classifier = nn.Identity()
            self.classifier = nn.Sequential(nn.Linear(in_features=768, out_features=768), nn.Tanh(), nn.Linear(in_features=768, out_features=output_shape[1]))
            self.model.config.drop_path_rate = 0
            fix_stack = None

        elif weight[:7] == 'roberta':
            configuration = AutoConfig.from_pretrained('roberta-base')
            self.model = AutoModel.from_pretrained("roberta-base", config = configuration)
           
            fix_stack = None if len(weight) == 7 else int(weight[7:])

        elif weight[:4] == 'bert':
            configuration = AutoConfig.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained("bert-base-uncased", config = configuration)
            fix_stack = None if len(weight) == 4 else int(weight[4:])

        if output_shape[1] == 88: # music
            self.dense = True
            
        set_grad_state(self.model, False)
        self.use_embedder = use_embedder

        if use_embedder:
            self.embedder = Embeddings1D(input_shape, output_shape, config=self.model.config, embed_dim=96 if weight == 'swin' else 768, target_seq_len=1024 if weight == 'swin' else 512, dense=self.dense, fix_stack=fix_stack)
            embedder_init(sself.model.swin.embeddings if weight == 'swin' else self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)    
            if weight == 'swin':
                self.model.swin.embeddings = self.embedder   
        else:
            self.embedder = nn.Identity()


        if not weight == 'swin': 
            if self.dense:
                self.model.embeddings = embedder_placeholder()
                self.model.pooler = nn.Identity()
                self.classifier = nn.Sequential(nn.Linear(in_features=768, out_features=output_shape[1]), nn.Sigmoid())
            else:
                self.model.embeddings = embedder_placeholder()
                self.model.pooler = nn.Identity()
                self.classifier = nn.Linear(in_features=768, out_features=output_shape[1])     


    def forward(self, x):
        if self.weight == 'swin':
            if self.output_raw:
                return self.model.swin.embeddings(x)[0]

            x = self.model(x).logits
            return self.classifier(x)

        if self.output_raw:
            return self.embedder(x)
            
        x = self.embedder(x)

        if self.dense:
            x = self.model(inputs_embeds=x)['last_hidden_state']
        else:
            x = self.model(inputs_embeds=x)['pooler_output'].mean(1)

        return self.classifier(x)


class Embeddings2D(nn.Module):

    def __init__(self, input_shape, output_shape, patch_size=4, embed_dim=96, img_size=224, config=None):
        super().__init__()
        self.resize, self.input_dimensions = transforms.Resize((img_size, img_size)), (img_size, img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patched_dimensions = (self.input_dimensions[0] // self.patch_size[0], self.input_dimensions[1] // self.patch_size[1])
        ks = self.patch_size
        self.projection = nn.Conv2d(input_shape[1], embed_dim, kernel_size=ks, stride=self.patch_size, padding=(ks[0]-self.patch_size[0]) // 2)
        self.norm = nn.LayerNorm(embed_dim)
        num_patches = (self.input_dimensions[1] // self.patch_size[1]) * (self.input_dimensions[0] // self.patch_size[0])
        
        conv_init(self.projection)

        
    def maybe_pad(self, x, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            x = nn.functional.pad(x, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            x = nn.functional.pad(x, pad_values)
        return x


    def forward(self, x, *args, **kwargs):
        x = self.resize(x)
        _, _, height, width = x.shape

        x = self.maybe_pad(x, height, width)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, self.patched_dimensions


class Embeddings1D(nn.Module):
    def __init__(self, input_shape, output_shape, embed_dim=768, target_seq_len=512, config=None, dense=False, fix_stack=None):
        super().__init__()
        self.dense = dense
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len) if fix_stack is None else fix_stack
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(512, embed_dim, padding_idx=self.padding_idx)

        if self.dense:
            self.projection = nn.Linear(input_shape[-1], embed_dim)
        else:
            self.projection = nn.Conv1d(input_shape[1], embed_dim, kernel_size=self.stack_num, stride=self.stack_num)
        conv_init(self.projection)


    def get_stack_num(self, input_len, target_seq_len):
        if self.embed_dim == 768:
            for i in range(1, input_len + 1):
                if input_len % i == 0 and input_len // i <= target_seq_len:
                    break
            return i
        else:
            for i in range(1, input_len + 1):
                root = np.sqrt(input_len // i)
                if input_len % i == 0 and input_len // i <= target_seq_len and int(root + 0.5) ** 2 == (input_len // i):
                    break
            return i


    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is None:
            x = inputs_embeds
        b, c, l = x.shape

        x = self.projection(x)
        if not self.dense:
            x = x.transpose(1, 2)
        x = self.norm(x)

        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        position_embeddings = self.position_embeddings(position_ids)

        x = x + position_embeddings

        if self.embed_dim == 768:
            return x
        else:
            return x, self.patched_dimensions


def set_param_grad(model, finetune_method):

    if finetune_method == "layernorm":
        for n, m in model.named_parameters():
            if 'layer' in n:
                if 'layernorm' in n or 'LayerNorm' in n:
                    continue
                else:
                    m.requires_grad = False

    elif finetune_method == "non-attn":
        for n, m in model.named_parameters():
            if 'layer' in n:
                if 'query' in n or 'key' in n or 'value' in n:
                    m.requires_grad = False


####################################################

def get_tgt_model(args, root, sample_shape, num_classes, loss, add_loss=False, use_determined=False, context=None):
    
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    if len(sample_shape) == 4:
        IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
            
        src_model = wrapper2D(sample_shape, (1, num_classes), use_embedder=False, weight=args.weight, train_epoch=args.ep_tune_start)
        src_model = src_model.to(args.device).eval()
            
        src_feats = []
        src_ys = []
        for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            out = src_model(x_)
            if len(out.shape) > 2:
                out = out.mean(1)
            src_ys.append(y_.detach().cpu())
            src_feats.append(out.detach().cpu())
        src_feats = torch.cat(src_feats, 0)
        src_ys = torch.cat(src_ys, 0).long()
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)        
        del src_model    

    else:
        src_feats, src_ys = src_train_loader.dataset.tensors[0], src_train_loader.dataset.tensors[1]
        src_feats = src_feats.mean(1)
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        
    print("src feat shape", src_feats.shape, src_ys.shape) 
        
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None

    if args.dataset == 'PSICOV' or args.dataset == 'DARCY-FLOW-5' or args.dataset == 'DEEPSEA' or args.dataset == 'COSMIC' or args.dataset[:5] == 'MUSIC' or args.dataset == 'HOMOLOGY' or args.dataset == 'FSD':
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
    else:
        num_classes_new = num_classes

    tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)

    wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    tgt_model = wrapper_func(sample_shape, (1, num_classes), weight=args.weight, train_epoch=args.ep_tune_start)
    tgt_model = tgt_model.to(args.device).train()

    params_to_update = get_params_to_update(tgt_model, "")

    args.encoder_optimizer.params['lr'] *= 10
    tgt_model_optimizer = get_optimizer(args.encoder_optimizer.name, args.encoder_optimizer.params)(params_to_update)
    lr_lambda, _ = get_scheduler(args.encoder_scheduler.name, args.encoder_scheduler.params, args.ep_tune_start, 1)
    tgt_model_scheduler = torch.optim.lr_scheduler.LambdaLR(tgt_model_optimizer, lr_lambda=lr_lambda)
    tgt_model_optimizer.zero_grad()

    if args.objective == 'otdd-exact':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    elif args.objective == 'otdd-gaussian':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=False)
    
    # train tgt embedder
    score = 0
    reals, total_losses, times, together = [], [], [], []
    
    for ep in range(args.ep_tune_start + 1):   
        if args.ep_tune_start == 0: break

        total_loss = 0    
        time_start = default_timer()

        for i in np.random.permutation(num_classes_new):
            feats = []
            datanum = 0

            for j, data in enumerate(tgt_train_loaders[i]):
                
                if transform is not None:
                    x, y, z = data
                else:
                    x, y = data 
                
                x = x.to(args.device)
                out = tgt_model(x)
                feats.append(out)
                datanum += x.shape[0]
                
                if datanum > args.maxsamples: break

            feats = torch.cat(feats, 0)
            loss = tgt_class_weights[i] * score_func(feats)

            loss.backward()
            total_loss += loss.item()

        time_end = default_timer()  
        times.append(time_end - time_start) 

        feats = []
        ys = []
        datanum = 0

        for j, data in enumerate(tgt_train_loader):

            if transform is not None:
                x, y, z = data
            else:
                x, y = data 

            x = x.to(args.device)
            out = tgt_model(x)
            feats.append(out.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            datanum += x.shape[0]
                
            if datanum > 50 * args.maxsamples: break

        feats = np.concatenate(feats, 0)
        ys = np.concatenate(ys, 0)

        real = otdd(feats, ys=ys, src_train_dataset=src_train_dataset, exact=args.objective=='otdd-exact').item()
        real = float( "%.4f" % real)

        total_losses.append(total_loss)
        reals.append(real)
        together.append([total_losses[-1], reals[-1], times[-1]])

        if ep != 0:
            tgt_model_optimizer.step()
            tgt_model_scheduler.step()

        tgt_model_optimizer.zero_grad()

    if use_determined and len(reals) > 0:
        offset = np.random.randint(10)
        for ep, r in enumerate(reals):
            context.train.report_training_metrics(steps_completed=max(0, (ep + 1) * n_train - 1 - offset), metrics={"total loss": total_losses[ep], "real": r, "embed time": times[ep]})

    del tgt_train_loader, tgt_train_loaders
    torch.cuda.empty_cache()

    set_grad_state(tgt_model, True)

    if args.freeze:
        set_grad_state(tgt_model.embedder, False)
    tgt_model.output_raw = False

    set_param_grad(tgt_model, args.finetune_method)

    return tgt_model, get_params_to_update(tgt_model, ""), together



def main(use_determined, args, info=None, context=None):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = '/datasets' if use_determined else './datasets'

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)

    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    dims, sample_shape, num_classes, loss, config_kwargs = get_config(args.dataset)

    if load_embedder(use_determined, args):
        args.ep_tune_start = 0

    model, params_to_update, together = get_tgt_model(args, root, sample_shape, num_classes, loss, False, use_determined, context)
        
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split)

    metric, compare_metrics = get_metric(root, args.dataset)
    
    model, ep_start, id_best, train_score, train_losses, together_saved = load_state(use_determined, args, context, model, None, None, n_train, freq=args.validation_freq, test=True)
    together = together if together_saved is None else together_saved
    
    offset = 0 if ep_start == 0 else 1
    optimizer = get_optimizer(args.optimizer.name, args.optimizer.params)(params_to_update)
    lr_lambda, lr_sched_iter_ = get_scheduler(args.scheduler.name, args.scheduler.params, args.epochs, n_train)
    args.lr_sched_iter = lr_sched_iter_
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    n_train_temp = n_train

    if args.device == 'cuda':
        model.cuda()
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()

    # print(model)

    print("\n------- Experiment Summary --------")
    print("id:", args.experiment_id)
    print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer.params.lr)
    print("arch configs:", config_kwargs)
    print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
    print("finetune method:", args.finetune_method)
    print("param count:", count_params(model), count_trainable_params(model))
    
    model, ep_start, id_best, train_score, train_losses, together_saved = load_state(use_determined, args, context, model, optimizer, scheduler, n_train, freq=args.validation_freq)
    together = together if together_saved is None else together_saved
    train_time = []
    
    if ep_start == 0:
        print("\n------- Start Training --------")
    else:
        print("\n------- Resume Training --------")

    for ep in range(ep_start, args.epochs):
        time_start = default_timer()

        train_loss = train_one_epoch(context, args, model, optimizer, scheduler, train_loader, loss, n_train_temp, decoder, transform)
        train_time_ep = default_timer() -  time_start 

        if ep % args.validation_freq == 0 or ep == args.epochs - 1: 
                
            val_loss, val_score = evaluate(context, args, model, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)

            train_losses.append(train_loss)
            train_score.append(val_score)
            train_time.append(train_time_ep)

            print("[train", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))

            if use_determined:
                id_current = save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, together)
                context.train.report_training_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"train loss": train_loss, "epoch time": train_time_ep})
                context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"val score": val_score})
                    
            if compare_metrics(train_score) == val_score:
                if not use_determined:
                    id_current = save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, together)
                id_best = id_current
            
        if ep == args.epochs - 1:
            print("\n------- Start Test --------")
            test_scores = []
            test_model = model
            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            
            test_model, _, _, _, _, _ = load_state(use_determined, args, context, test_model, optimizer, scheduler, n_train, id_best, test=True)
            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            
            if use_determined:
                checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
                with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
                    np.save(os.path.join(path, 'test_score.npy'), test_scores)
            else:
                path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
                np.save(os.path.join(path, 'test_score.npy'), test_scores)

            if len(together) > 0:
                print(together[0], together[-1])

        if use_determined and context.preempt.should_preempt():
            print("paused")
            return


def train_one_epoch(context, args, model, optimizer, scheduler, loader, loss, temp, decoder=None, transform=None, debug_mode=False):

    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):

        if transform is not None:
            x, y, z = data
            z = z.to(args.device)
        else:
            x, y = data 
        
        x, y = x.to(args.device), y.to(args.device)
        out = model(x)

        if isinstance(out, dict):
            out = out['out']

        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)

        if transform is not None:
            out = transform(out, z)
            y = transform(y, z)
        
        try:
            if y.shape[1] == 1: y = y.squeeze()         
        except:
            pass 

        if args.dataset == 'MUSIC':
            l = loss(out, y) * y.shape[1]
        else:
            l = loss(out, y)
        l.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if (i + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if args.lr_sched_iter:
            scheduler.step()

        train_loss += l.item()

        if i >= temp - 1:
            break

    if (not args.lr_sched_iter):
        scheduler.step()

    return train_loss / temp


def evaluate(context, args, model, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
    model.eval()
    
    eval_loss, eval_score = 0, 0
    
    if fsd_epoch is None:

        ys, outs, n_eval = [], [], 0
        with torch.no_grad():
            for data in loader:
                if transform is not None:
                    x, y, z = data
                    z = z.to(args.device)
                else:
                    x, y = data
                                    
                x, y = x.to(args.device), y.to(args.device)

                if args.dataset == 'PSICOV':
                    out_is = []
                    for i in range(4):
                        out_js = []
                        for j in range(4):
                            x_ = x[..., i * 128:(i+1)*128, j * 128:(j+1)*128]
                            out_ = model(x_)
                            out_js.append(out_)
                        out_is.append(torch.cat(out_js, dim=-1))
                    out = torch.cat(out_is, dim=-2)
                else:
                    out = model(x)

                if isinstance(out, dict):
                    out = out['out']

                if decoder is not None:
                    out = decoder.decode(out).view(x.shape[0], -1)
                    y = decoder.decode(y).view(x.shape[0], -1)
                                    
                if transform is not None:
                    out = transform(out, z)
                    y = transform(y, z)

                outs.append(out)
                ys.append(y)

                if len(ys) * y.shape[0] >= args.eval_batch_size:
                    outs = torch.cat(outs, 0)
                    ys = torch.cat(ys, 0)

                    try:
                        eval_loss += loss(outs, ys.squeeze()).item()
                        eval_score += metric(outs, ys).item()
                        n_eval += 1
                    except:
                        pass
                    
                    ys, outs = [], []

            if n_eval == 0:
                outs = torch.cat(outs, 0)
                ys = torch.cat(ys, 0)
                eval_loss = loss(outs, ys.squeeze()).item()
                eval_score = metric(outs, ys).item()
            else:
                eval_loss /= n_eval
                eval_score /= n_eval

    else:
        outs, ys = [], []
        with torch.no_grad():
            for ix in range(loader.len):

                if fsd_epoch < 100:
                    if ix > 2000: break

                x, y = loader[ix]
                x, y = x.to(args.device), y.to(args.device)
                out = model(x).mean(0).unsqueeze(0)
                eval_loss += loss(out, y).item()
                outs.append(torch.sigmoid(out).detach().cpu().numpy()[0])
                ys.append(y.detach().cpu().numpy()[0])

        outs = np.asarray(outs).astype('float32')
        ys = np.asarray(ys).astype('int32')
        stats = calculate_stats(outs, ys)
        eval_score = np.mean([stat['AP'] for stat in stats])
        eval_loss /= n_eval

    return eval_loss, eval_score


########################## Helper Funcs ##########################

def save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, together):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, together)
        return ep

    else:
        checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
        with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
            save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, together)
            return uuid


def save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, together):
    np.save(os.path.join(path, 'hparams.npy'), args)
    np.save(os.path.join(path, 'train_score.npy'), train_score)
    np.save(os.path.join(path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(path, 'pretrain_stats.npy'), together)

    model_state_dict = {
                'network_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
    torch.save(model_state_dict, os.path.join(path, 'state_dict.pt'))

    rng_state_dict = {
                'cpu_rng_state': torch.get_rng_state(),
                'gpu_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate()
            }
    torch.save(rng_state_dict, os.path.join(path, 'rng_state.ckpt'))


def load_embedder(use_determined, args):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        return os.path.isfile(os.path.join(path, 'state_dict.pt'))
    else:

        info = det.get_cluster_info()
        checkpoint_id = info.latest_checkpoint
        return checkpoint_id is not None


def load_state(use_determined, args, context, model, optimizer, scheduler, n_train, checkpoint_id=None, test=False, freq=1):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
            return model, 0, 0, [], [], None
    else:

        if checkpoint_id is None:
            info = det.get_cluster_info()
            checkpoint_id = info.latest_checkpoint
            if checkpoint_id is None:
                return model, 0, 0, [], [], None
        
        checkpoint = client.get_checkpoint(checkpoint_id)
        path = checkpoint.download()

    train_score = np.load(os.path.join(path, 'train_score.npy'))
    train_losses = np.load(os.path.join(path, 'train_losses.npy'))
    together = np.load(os.path.join(path, 'pretrain_stats.npy'))
    epochs = freq * (len(train_score) - 1) + 1
    checkpoint_id = checkpoint_id if use_determined else epochs - 1
    model_state_dict = torch.load(os.path.join(path, 'state_dict.pt'))
    model.load_state_dict(model_state_dict['network_state_dict'])
    
    if not test:
        optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(model_state_dict['scheduler_state_dict'])

        rng_state_dict = torch.load(os.path.join(path, 'rng_state.ckpt'), map_location='cpu')
        torch.set_rng_state(rng_state_dict['cpu_rng_state'])
        torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
        np.random.set_state(rng_state_dict['numpy_rng_state'])
        random.setstate(rng_state_dict['py_rng_state'])

        if use_determined: 
            try:
                for ep in range(epochs):
                    if ep % freq == 0:
                        context.train.report_training_metrics(steps_completed=(ep + 1) * n_train, metrics={"train loss": train_losses[ep // freq]})
                        context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train, metrics={"val score": train_score[ep // freq]})
            except:
                print("load error")

    return model, epochs, checkpoint_id, list(train_score), list(train_losses), together


def infer_labels(loader, k = 10):
    from sklearn.cluster import k_means, MiniBatchKMeans
    
    X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
    try:
        Z = loader.dataset.tensors[2].cpu()
    except:
        Z = None

    Y = Y.reshape(len(Y), -1)

    if len(Y) <= 10000:
        labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
        Y = labeling_fun(Y).unsqueeze(1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
        Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

    if Z is None:
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k


def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}

    if len(train_set.__getitem__(0)) == 3:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
    class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
    return loaders, class_weights


def create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx=1):
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
    return position_ids.unsqueeze(0).expand(input_shape)


def scale(Z, numpy=False):
    if numpy:
        factor = np.trace(Z.transpose() @ Z)
        return Z / np.sqrt(factor)
    
    factor = torch.trace(Z.transpose(0, 1) @ Z)
    return Z / torch.sqrt(factor)


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def set_grad_state(module, state):
    for n, m in module.named_modules():
        if len(n) == 0: continue
        if not state and 'position' in n: continue
        if not state and 'tunable' in n: continue
        for param in m.parameters():
            param.requires_grad = state



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')

    args = parser.parse_args()
    if args.config is not None:     
        import yaml

        with open(args.config, 'r') as stream:
            args = AttrDict(yaml.safe_load(stream)['hyperparameters'])
            main(False, args)

    else:
        import determined as det
        from determined.experimental import client
        from determined.pytorch import DataLoader

        info = det.get_cluster_info()
        args = AttrDict(info.trial.hparams)
        
        with det.core.init() as context:
            main(True, args, info, context)

