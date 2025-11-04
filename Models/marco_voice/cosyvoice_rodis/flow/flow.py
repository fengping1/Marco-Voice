# 

import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice_rodis.utils.mask import make_pad_mask
from cosyvoice_rodis.utils.losses import OrthogonalityLoss

class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000},
                 flow_emotion_embedding: bool = False,  
                 flow_orth_loss: bool = False,
                 cross_orth_loss: bool = False):  

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss
        self.flow_emotion_embedding = flow_emotion_embedding
        self.flow_orth_loss = flow_orth_loss
        self.cross_orth_loss = cross_orth_loss

        if self.flow_emotion_embedding:
            self.flow_emotion_embedding_proj = torch.nn.Linear(spk_embed_dim, spk_embed_dim)
            self.speaker_projector = nn.Linear(spk_embed_dim, spk_embed_dim)

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)  
        token_len = batch['speech_token_len'].to(device) 
        feat = batch['speech_feat'].to(device) 
        feat_len = batch['speech_feat_len'].to(device) 
        embedding = batch['embedding'].to(device) 


        if self.flow_emotion_embedding:
            flow_emotion_embedding = batch['emotion_embedding'].to(device) #[270,192]
            flow_emotion_embedding = F.normalize(flow_emotion_embedding, dim=1)
            flow_emotion_embedding = self.flow_emotion_embedding_proj(flow_emotion_embedding)
            embedding = self.speaker_projector(embedding)
            
            if self.cross_orth_loss:  #false
                print("进")  
                orth_loss = 0.0            
                batch_size = embedding.size(0)
                if batch_size > 1:
                    batch_contrastive_orth_loss=0
                    for i in range(batch_size):
                        for j in range(i + 1, batch_size):
                            
                            batch_contrastive_orth_loss =batch_contrastive_orth_loss+ torch.abs(torch.dot(embedding[i], flow_emotion_embedding[j])) 
                    num_pairs = (batch_size * (batch_size - 1)) / 2  
                    batch_contrastive_orth_loss = batch_contrastive_orth_loss / num_pairs
                else:
                    batch_contrastive_orth_loss = 0
                orth_loss = OrthogonalityLoss(embedding, flow_emotion_embedding)
            else:
                orth_loss = OrthogonalityLoss(embedding, flow_emotion_embedding)
                

        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding) 

        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device) 
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  

        # text encode
        h, h_lengths = self.encoder(token, token_len, flow_emotion_embedding) 
        h = self.encoder_proj(h) 
        h, h_lengths = self.length_regulator(h, feat_len) 
        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)  

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1) 
        mse_loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(), 
            embedding,  
            cond=conds  
        ) 
        if self.flow_orth_loss and self.flow_emotion_embedding:

            loss =mse_loss+ orth_loss+batch_contrastive_orth_loss

        return {'loss': loss, "mse_loss":mse_loss,"orth_loss":orth_loss,"contrastive_loss":batch_contrastive_orth_loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache,
                  flow_emotion_embedding=None):  
        assert token.shape[0] == 1
        if self.flow_emotion_embedding and flow_emotion_embedding is not None: 
            flow_emotion_embedding = F.normalize(flow_emotion_embedding.unsqueeze(0).to(torch.float16), dim=1)
            flow_emotion_embedding = self.flow_emotion_embedding_proj(flow_emotion_embedding) 
            embedding = self.speaker_projector(embedding)
            embedding += flow_emotion_embedding  

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1] 
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len 
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  

        # text encode
        h, h_lengths = self.encoder(token, token_len, flow_emotion_embedding)
        h = self.encoder_proj(h) #torch.Size([1, 358, 80])
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256) 
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate) 

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device) 
        conds[:, :mel_len1] = prompt_feat 
        conds = conds.transpose(1, 2) 

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(), 
            mask=mask.unsqueeze(1), 
            spks=embedding, 
            cond=conds, 
            n_timesteps=10,
            prompt_len=mel_len1, 
            flow_cache=flow_cache 
        ) 
        feat = feat[:, :, mel_len1:] 
        assert feat.shape[2] == mel_len2
        return feat, flow_cache