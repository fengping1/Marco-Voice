# 

import torch
import torch.nn.functional as F

def tpr_loss(disc_real_outputs, disc_generated_outputs, tau):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        m_DG = torch.median((dr - dg))
        L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def mel_loss(real_speech, generated_speech, mel_transforms):
    loss = 0
    for transform in mel_transforms:
        mel_r = transform(real_speech)
        mel_g = transform(generated_speech)
        loss += F.l1_loss(mel_g, mel_r)
    return loss

def OrthogonalityLos_old(speaker_embedding, emotion_embedding):  #speaker_embedding[270,192] emotion_embedding[270,192]
        #print("OrthogonalityLoss")
        speaker_embedding_t = speaker_embedding.t() #[192,270]
        dot_product_matrix = torch.matmul(emotion_embedding, speaker_embedding_t)#[270,270]
        emotion_norms = torch.norm(emotion_embedding, dim=1, keepdim=True) #[270,1]
        speaker_norms = torch.norm(speaker_embedding, dim=1, keepdim=True).t() #[1,270]
        normalized_dot_product_matrix = dot_product_matrix / (emotion_norms * speaker_norms) #[270,270]
        ort_loss = torch.norm(normalized_dot_product_matrix, p='fro')**2 #说话人与情感正交
        cosine_sim = F.cosine_similarity(emotion_embedding.unsqueeze(2), speaker_embedding.unsqueeze(1), dim=-1) #【270，192，1】与[270, 1, 192]在最后一个维度计算cos相似度 cosine_sim形状是[270, 192] 
        cosine_ort_loss = torch.norm(cosine_sim.mean(dim=-1), p='fro') ** 2

        #return  0.01 * (ort_loss + cosine_ort_loss) #ort_loss是2292.1016 cosine_ort_loss是0.0554 直接加是否合理？  目的就是是的情感与说话人正交？ 那么就是分离解耦？
        return  1 * (ort_loss * 0.001 + cosine_ort_loss*10)

        #flow时候 ort_loss 2749.5303  cosine_ort_loss 0.0005

def OrthogonalityLoss(speaker_embedding, emotion_embedding):  #speaker_embedding[270,192] emotion_embedding[270,192]
        #print("OrthogonalityLoss")
        speaker_embedding_t = speaker_embedding.t() #[192,270]
        dot_product_matrix = torch.matmul(emotion_embedding, speaker_embedding_t)#[270,270]
        emotion_norms = torch.norm(emotion_embedding, dim=1, keepdim=True) #[270,1]
        speaker_norms = torch.norm(speaker_embedding, dim=1, keepdim=True).t() #[1,270]
        normalized_dot_product_matrix = dot_product_matrix / (emotion_norms * speaker_norms) #[270,270]
        ort_loss = torch.norm(normalized_dot_product_matrix, p='fro')**2 #说话人与情感正交
        cosine_sim = F.cosine_similarity(emotion_embedding.unsqueeze(2), speaker_embedding.unsqueeze(1), dim=-1) #【270，192，1】与[270, 1, 192]在最后一个维度计算cos相似度 cosine_sim形状是[270, 192] 
        cosine_ort_loss = torch.norm(cosine_sim.mean(dim=-1), p='fro') ** 2

        return  0.01 * (ort_loss + cosine_ort_loss) #ort_loss是2292.1016 cosine_ort_loss是0.0554 直接加是否合理？  目的就是是的情感与说话人正交？ 那么就是分离解耦？
        #return  1 * (ort_loss * 0.001 + cosine_ort_loss*10)

        #flow时候 ort_loss 2749.5303  cosine_ort_loss 0.0005

def ContrastiveOrthogonalLoss(spk_pure, emo_pure, temperature=0.1):
    # spk_pure, emo_pure: [B, D]
    spk_pure = F.normalize(spk_pure, dim=-1)
    emo_pure = F.normalize(emo_pure, dim=-1)

    # 相似度矩阵
    sim_matrix = torch.mm(spk_pure, emo_pure.T) / temperature  # [B, B]

    # 对角线是正样本对（同一个样本的 speaker vs emotion），但我们希望它们不相似！
    # 所以我们最小化正样本对得分，最大化负样本对得分 → 反向 InfoNCE
    labels = torch.arange(sim_matrix.size(0)).to(spk_pure.device)
    loss = F.cross_entropy(-sim_matrix, labels)  # 负号：让正样本对得分低
    return loss