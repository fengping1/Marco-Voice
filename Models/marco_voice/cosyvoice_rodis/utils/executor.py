# 

# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist

from cosyvoice_rodis.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join

class Executor:

    def __init__(self, gan: bool = False):
        self.gan = gan
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx

                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice_rodis.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def train_one_epoc_new(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join):

        import torch
        import logging
        import traceback
        from contextlib import nullcontext

        rank = self.rank
        try:
            lr = optimizer.param_groups[0]['lr']
            logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, rank))
            logging.info('using accumulate grad, new batch size is {} times larger than before'.format(info_dict['accum_grad']))

            # ✅ 调试：打印当前 rank 开始训练
            print(f"🟢 RANK {rank} STARTING EPOCH {self.epoch}")

            model.train()
            model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
            with model_context():
                for batch_idx, batch_dict in enumerate(train_data_loader):
                    try:
                        info_dict["tag"] = "TRAIN"
                        info_dict["step"] = self.step
                        info_dict["epoch"] = self.epoch
                        info_dict["batch_idx"] = batch_idx

                        # ✅ 调试：每 50 个 batch 打印一次（避免日志太多）
                        if batch_idx % 50 == 0:
                            print(f"🟢 RANK {rank} | Epoch {self.epoch} | Batch {batch_idx} | Step {self.step}")

                        if cosyvoice_join(group_join, info_dict):
                            break

                        # Disable gradient synchronizations across DDP processes.
                        if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                            context = model.no_sync
                        else:
                            context = nullcontext

                        with context():
                            info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                            info_dict = batch_backward(model, scaler, info_dict)

                        info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)

                        # ✅ 调试：检查 loss 是否正常
                        loss = info_dict.get("loss", None)
                        if loss is not None:
                            if torch.isfinite(loss).all() == False:
                                print(f"❌ RANK {rank} | Epoch {self.epoch} | Batch {batch_idx} | ❌ LOSS is NaN or Inf: {loss.item()}")
                                raise ValueError(f"Loss value is invalid: {loss.item()}")

                        # ✅ 调试：检查 grad_norm 是否正常
                        if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                            grad_norm = info_dict.get("grad_norm", None)
                            if grad_norm is not None:
                                if not (float('inf') > grad_norm > float('-inf')) or grad_norm != grad_norm:  # nan check
                                    print(f"❌ RANK {rank} | Epoch {self.epoch} | Batch {batch_idx} | ❌ GRAD NORM is NaN or Inf: {grad_norm}")
                                    raise ValueError(f"grad_norm is invalid: {grad_norm}")

                        log_per_step(writer, info_dict)

                        # ✅ 调试：每 100 个 batch 同步一次 GPU 和分布式进程
                        if batch_idx % 100 == 0:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()  # 确保 GPU 操作完成
                            try:
                                dist.barrier(group_join)
                                if rank == 0:
                                    print(f"✅ ALL RANKS SYNCED at Epoch {self.epoch}, Batch {batch_idx}")
                            except Exception as e:
                                print(f"❌ Barrier failed at Epoch {self.epoch}, Batch {batch_idx}, Rank {rank}: {e}")
                                raise

                        # NOTE specify save_per_step in config if you want to enable step save
                        if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                        (batch_idx + 1) % info_dict["accum_grad"] == 0:
                            dist.barrier()
                            self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                            model.train()

                        if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                            self.step += 1

                    except Exception as e:
                        # ✅ 关键：捕获 batch 级异常并打印详细信息
                        print(f"❌ CRITICAL ERROR in RANK {rank}, EPOCH {self.epoch}, BATCH {batch_idx}, STEP {self.step}")
                        print(f"Exception Type: {type(e).__name__}")
                        print(f"Error Message: {e}")
                        print("📜 Traceback:")
                        print("".join(traceback.format_tb(e.__traceback__)))
                        raise  # 重新抛出，让外层也感知到

            # ✅ 调试：epoch 结束
            dist.barrier()
            self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)
            print(f"✅ RANK {rank} FINISHED EPOCH {self.epoch}")

        except Exception as e:
            # ✅ 最外层捕获
            print(f"❌ FATAL ERROR in RANK {rank} during EPOCH {self.epoch}")
            print(f"Exception: {type(e).__name__}: {e}")
            print("📜 Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            raise

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice_rodis.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                #total_loss_dict[k].append(v.item() * num_utts)
                value = v.item() if hasattr(v, 'item') else v
                total_loss_dict[k].append(value * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)
