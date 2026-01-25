import logging
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from utils import ensure_dir,set_color,get_local_time,delete_file
import os
import heapq
import random
import ot
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def random_print(cont, p=0.1):
    if random.random()<p:
        print(cont)

# a*cf+(1-a)*stable
class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        if args.dir=="":
            saved_model_dir = "{}".format(get_local_time())
        else:
            saved_model_dir = args.dir
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

        self.cf_loss = args.cf_loss
        if self.cf_loss:
            # 加载协同向量
            self.cf_emb=torch.load(args.cf_path, map_location=self.device)
            self.cf_emb=self.cf_emb.weight

        # finetune过程中减少码字漂移的约束
        args = vars(args)
        self.stable_loss = args.get("stable_loss", 0)
        self.kl_loss_all = args.get("kl_loss_all", 0)
        self.kl_temp = args.get("kl_temp", 1)
        self.kl_weight = args.get("kl_weight", 5)
        self.loss_old_weight = args.get("loss_old", 0.1)
        self.topk = args.get("topk", 0.1)
        self.stable_weight = args.get("stable_weight", 0)
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"./runs/{time_str}"
        self.writer = SummaryWriter(log_dir=log_dir)
       
    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1] # 0是向量，1是物品编号
            data = data.to(self.device)
            self.optimizer.zero_grad()
            indices=self.old_book[(emb_idx-1).detach().cpu()]
            indices=torch.tensor(indices,dtype=torch.long,device=self.device)
            target_old = self.old_encoded_x[emb_idx-1]
            current_cf = self.cf_emb[emb_idx]
            out, rq_loss, indices, dense_out, oldxe, old_dense_out, gate_score, old_rq_loss, old_out = self.model(data, current_cf=current_cf, target_old=target_old, writer=self.writer, epoch_idx=epoch_idx)
            loss_new, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            loss_old, old_loss_recon = self.model.compute_loss(old_out, old_rq_loss, xs=data)
            origin_loss=loss_new.item()
            emb_idx=torch.tensor(emb_idx, dtype=torch.long, device=self.device)
            self.writer.add_scalar('Train/Loss_recon', loss_recon.item(), epoch_idx)
            # 增加CF loss
            if self.cf_loss:
                cf_loss_new=self.model.compute_cf_loss(emb_idx, dense_out, self.cf_emb,)
                cf_loss_old=self.model.compute_cf_loss(emb_idx, old_dense_out, self.cf_emb,)
                loss_new += 0.02*cf_loss_new
                loss_old += 0.02*cf_loss_old
                origin_loss+=0.02*cf_loss_new.item() 
                self.writer.add_scalar('Train/Loss_CF_new', cf_loss_new.item(), epoch_idx)
                self.writer.add_scalar('Train/Loss_CF_old', cf_loss_old.item(), epoch_idx)
            
            if self.stable_loss:
                stable_loss=self.stable_weight*self.model.compute_stable_loss(oldxe, emb_idx, self.old_encoded_x)
                loss_old+=stable_loss
                self.writer.add_scalar('Train/stable_loss', stable_loss.item(), epoch_idx)
        
            loss = 0
            
            if self.kl_loss_all:
                kl_xe = self.model.encoder(data)
                student_log_probs = self.model.get_layer_log_probs(kl_xe, temperature=self.kl_temp)
                teacher_log_probs = self.teacher_log_probs[emb_idx-1].detach()
                kl_loss = F.kl_div(
                    student_log_probs, 
                    teacher_log_probs, 
                    reduction='none', 
                    log_target=True
                ).mean(dim=-1)
                kl_loss=kl_loss.mean()
                loss += self.kl_weight*kl_loss
                self.writer.add_scalar('Train/kl_loss_all', kl_loss.item(), epoch_idx)

            loss += loss_new
            loss += self.loss_old_weight*loss_old
            gate_loss = (gate_score**2).mean()
            loss += 0.001*gate_loss
            self.writer.add_scalar('Train/gate_loss', gate_loss.item(), epoch_idx)

            self._check_nan(loss)
            loss.backward()
           
            self.writer.add_scalar('Train/Loss', loss.item(), epoch_idx)
            self.writer.add_scalar('Train/Loss_origin', origin_loss, epoch_idx)
            self.writer.add_scalar('Train/Loss_new', loss_new, epoch_idx)
            self.writer.add_scalar('Train/Loss_old', loss_old, epoch_idx)
            pre_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1] # 0是向量，1是物品编号
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"
    


    def set_oldbook(self, data):
        iter_data = tqdm(
                    data,
                    total=len(data),
                    ncols=100,
                    )
        self.model.eval()
        old_codebook=[]
        old_encodex=[]
        teacher_log_probs_list=[]
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)
            old_codebook.append(self.model.get_indices(data).detach().cpu().numpy())
            old_encodex.append(self.model.encoder(data).detach().cpu().numpy())
            xe = self.model.encoder(data)
            teacher_log_probs = self.model.get_layer_log_probs(xe, temperature=self.kl_temp)
            teacher_log_probs_list.append(teacher_log_probs.detach())
            
        self.old_book = np.concatenate(old_codebook, axis=0)  
        self.old_encoded_x = np.concatenate(old_encodex, axis=0)  
        self.old_encoded_x = torch.tensor(self.old_encoded_x, device=self.model.device)
        self.origin_book = np.concatenate(old_codebook, axis=0)  

        # 保存codebook
        self.origin_code = self.model.rq.get_codebook()
        self.teacher_log_probs=torch.concat(teacher_log_probs_list,dim=0)

    def indices_change(self, dl, epoch_idx):
        # 计算变化的码字数量，和self.origin_book做对比
        new_indices=[]
        for i, (item_emb, idx) in enumerate(dl):
            item_emb = torch.tensor(item_emb, device=self.model.device)
            
            with torch.no_grad():
                new_indice = self.model.get_indices(item_emb)   # [B, L]
            new_indices.append(new_indice.cpu())
        new_indices=np.concatenate(new_indices, axis=0)
        L = new_indices.shape[-1]
        origin = self.origin_book
        unchanged_per_layer = (new_indices == origin).sum(axis=0)  # shape: [L]
        all_unchanged = (new_indices == origin).all(axis=1)       # shape: [N], bool
        num_all_unchanged = all_unchanged.sum()                   # scalar

        # 转为 Python int 列表：L 个 per-layer + 1 个 all-layers
        result = unchanged_per_layer.tolist() + [int(num_all_unchanged)]
        for i, res in enumerate(result[:-1]):
            self.writer.add_scalar(f'Train/Layer{i+1}', res, epoch_idx)

        self.writer.add_scalar(f'Train/AllLayer', result[-1], epoch_idx)

        return result  


    def fit(self, data, data_noshuffle):

        cur_eval_step = 0
        
        # 准备旧索引；不从codebook加载，因为有部分不一样
        self.set_oldbook(data_noshuffle)
        log_path = os.path.join(self.ckpt_dir, "log.txt")
        with open(log_path, "w") as f:
            pass

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            if epoch_idx%10==0:
                change_nums = self.indices_change(data_noshuffle, epoch_idx) # 统计变化物品的数量
                
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)


            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1
                with open(log_path, 'a') as f:
                    f.write(f"Epoch: {epoch_idx},\tcollision_rate is {collision_rate}\n")


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
        
        ckpt_path = self._save_checkpoint(epoch_idx,ckpt_file="last.pth")


        return self.best_loss, self.best_collision_rate




