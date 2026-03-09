import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.logger import logger


class Crop_Dataset(Dataset):
    def __init__(self, crops, motion_ids, frame_indices, transform=None):
        self.crops = crops
        self.motion_ids = motion_ids
        self.frame_indices = frame_indices
        self.transform = transform

        self.frame_to_indices = {}
        for i, f_idx in enumerate(frame_indices):
            if f_idx not in self.frame_to_indices:
                self.frame_to_indices[f_idx] = []
            self.frame_to_indices[f_idx].append(i)
    
    def __len__(self):
        return len(self.crops)
    
    def __getitem__(self, idx):
        crop = self.crops[idx]
        if self.transform:
            crop = self.transform(crop)
        return crop, self.motion_ids[idx], self.frame_indices[idx], idx


class Identity_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    def forward(self, x):
        features = self.encoder(x)
        features = features.flatten(start_dim=1)  
        return F.normalize(self.projector(features), dim=1)


class Contrastive_Trainer:
    def __init__(self, is_ir:bool=False, slide_window:int=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = Identity_Encoder().to(device)
        self.slide_window = slide_window

        if is_ir:
            tr_nm_mean = [0.485, 0.485, 0.485]
            tr_nm_std = [0.229, 0.229, 0.229]
        else:
            tr_nm_mean = [0.485, 0.456, 0.406]
            tr_nm_std=[0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tr_nm_mean, std=tr_nm_std),
        ])
    
    def mine_triplets(self, dataset:Crop_Dataset, window:int=5, max_triplets:int=5000):
        triplets = []
        frames = sorted(dataset.frame_to_indices.keys())
        
        for frame_idx in frames:
            indices_in_frame = dataset.frame_to_indices[frame_idx]
            if len(indices_in_frame) != 2:
                continue
            
            idx_a, idx_b = indices_in_frame

            for delta in range(1, window + 1):
                next_frame = frame_idx + delta
                if next_frame in dataset.frame_to_indices:
                    next_indices = dataset.frame_to_indices[next_frame]
                    for next_idx in next_indices:
                        if dataset.motion_ids[next_idx] == dataset.motion_ids[idx_a]:
                            triplets.append((idx_a, next_idx, idx_b))
                            if len(triplets) >= max_triplets:
                                return triplets
                    break
    
        return triplets

    def mine_hard_triplets_adaptive(self, dataset:Crop_Dataset, embeddings, pos_threshold=0.5, neg_threshold=0.3, max_triplets=5000):
        sim_matrix = cosine_similarity(embeddings)
        triplets = []
        
        for i in range(len(embeddings)):
            anchor_id = dataset.motion_ids[i]
            frame_idx = dataset.frame_indices[i]

            same_mouse_mask = np.array(dataset.motion_ids) == anchor_id
            same_mouse_sims = sim_matrix[i].copy()
            same_mouse_sims[~same_mouse_mask] = 2.0
            same_mouse_sims[i] = 2.0

            frame_indices_arr = np.array(dataset.frame_indices)
            distant_mask = np.abs(frame_indices_arr - frame_idx) >= self.slide_window
            same_mouse_sims[distant_mask] = 2.0

            if np.any(same_mouse_mask):
                hard_pos_candidates = np.where(same_mouse_sims < pos_threshold)[0]
                if len(hard_pos_candidates) > 0:
                    hardest_pos_idx = hard_pos_candidates[np.argmin(same_mouse_sims[hard_pos_candidates])]
    
                    diff_mouse_mask = ~same_mouse_mask
                    diff_mouse_sims = sim_matrix[i].copy()
                    diff_mouse_sims[~diff_mouse_mask] = -2.0
                    diff_mouse_sims[distant_mask] = -2.0

                    if np.any(diff_mouse_mask):
                        hard_neg_candidates = np.where(diff_mouse_sims > neg_threshold)[0]
                        if len(hard_neg_candidates) > 0:
                            hardest_neg_idx = hard_neg_candidates[np.argmax(diff_mouse_sims[hard_neg_candidates])]
                            triplets.append((i, hardest_pos_idx, hardest_neg_idx))
            
            if len(triplets) >= max_triplets:
                break
        
        logger.info(f"[EMBHARD] Found {len(triplets)} hard triplets (pos<{pos_threshold:.2f}, neg>{neg_threshold:.2f})")
        return triplets

    def contrastive_loss_with_variance(self, anchor_emb, pos_emb, neg_emb, margin=0.5, var_weight=0.1):
        pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
        triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()

        all_embs = torch.cat([anchor_emb, pos_emb, neg_emb], dim=0)
        var_loss = torch.mean(torch.var(all_embs, dim=0))
        var_penalty = torch.abs(var_loss - 1.0)

        return triplet_loss + var_weight * var_penalty
        
    def train(
            self,
            dataset:Crop_Dataset,
            batch_size=64,
            lr=1e-5,
            epochs=20,
            warmup_epochs=10,
            hard_mine_interval=5,
            sil_check_interval=5,
            sil_threshold=0.4,
            sil_subsampling=1000,
            ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        logger.info("[CONTRAIN] Warm-up training with easy triplets...")
        easy_triplets = self.mine_triplets(dataset, window=5, max_triplets=5000)
        logger.info(f"[CONTRAIN] Mined {len(easy_triplets)} easy triplets")
        
        self.model.train()
        for epoch in range(warmup_epochs):
            total_loss = 0
            np.random.shuffle(easy_triplets)
    
            num_batches = max(1, len(easy_triplets) // batch_size)
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                batch_triplets = easy_triplets[start:end]
                
                anchor_imgs = []
                pos_imgs = []
                neg_imgs = []
                
                for anchor_idx, pos_idx, neg_idx in batch_triplets:
                    anchor_img, _, _, _ = dataset[anchor_idx]
                    pos_img, _, _, _ = dataset[pos_idx]
                    neg_img, _, _, _ = dataset[neg_idx]
                    
                    anchor_imgs.append(self.transform(anchor_img))
                    pos_imgs.append(self.transform(pos_img))
                    neg_imgs.append(self.transform(neg_img))
                
                anchor_batch = torch.stack(anchor_imgs).to(self.device)
                pos_batch = torch.stack(pos_imgs).to(self.device)
                neg_batch = torch.stack(neg_imgs).to(self.device)
                
                anchor_emb = self.model(anchor_batch)
                pos_emb = self.model(pos_batch)
                neg_emb = self.model(neg_batch)
                
                loss = self.contrastive_loss_with_variance(anchor_emb, pos_emb, neg_emb)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"[CONTRAIN] Epoch {epoch+1}/{warmup_epochs}, Loss: {total_loss/num_batches:.4f}")

        logger.info("[CONTRAIN] Mining hard triplets from trained embeddings...")
        self.model.eval()
        embeddings = self.extract_embeddings(dataset)
        sim_matrix = cosine_similarity(embeddings)
        
        same_mouse_sims = []
        diff_mouse_sims = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                if dataset.frame_indices[j] - dataset.frame_indices[i] > self.slide_window:
                    continue
                if dataset.motion_ids[i] == dataset.motion_ids[j]:
                    same_mouse_sims.append(sim_matrix[i, j])
                else:
                    diff_mouse_sims.append(sim_matrix[i, j])
        
        pos_threshold = np.percentile(same_mouse_sims, 20)
        neg_threshold = np.percentile(diff_mouse_sims, 80)
        
        logger.info(f"[CONTRAIN] Same-mouse: {np.mean(same_mouse_sims):.3f} ± {np.std(same_mouse_sims):.3f}")
        logger.info(f"[CONTRAIN] Diff-mouse: {np.mean(diff_mouse_sims):.3f} ± {np.std(diff_mouse_sims):.3f}")
        logger.info(f"[CONTRAIN] pos<{pos_threshold:.3f}, neg>{neg_threshold:.3f}")
        
        hard_triplets = self.mine_hard_triplets_adaptive(dataset, embeddings, 
                                                        pos_threshold=pos_threshold, 
                                                        neg_threshold=neg_threshold,
                                                        max_triplets=5000)
        logger.info(f"[CONTRAIN] Mined {len(hard_triplets)} hard triplets")

        logger.info("\n[CONTRAIN] Training on hard triplets...")
        self.model.train()
        for epoch in range(warmup_epochs, epochs):
            total_loss = 0
            np.random.shuffle(hard_triplets)

            num_batches = max(1, len(hard_triplets) // batch_size)
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                batch_triplets = hard_triplets[start:end]
                
                anchor_imgs = []
                pos_imgs = []
                neg_imgs = []
                
                for anchor_idx, pos_idx, neg_idx in batch_triplets:
                    anchor_img, _, _, _ = dataset[anchor_idx]
                    pos_img, _, _, _ = dataset[pos_idx]
                    neg_img, _, _, _ = dataset[neg_idx]
                    
                    anchor_imgs.append(self.transform(anchor_img))
                    pos_imgs.append(self.transform(pos_img))
                    neg_imgs.append(self.transform(neg_img))
                
                anchor_batch = torch.stack(anchor_imgs).to(self.device)
                pos_batch = torch.stack(pos_imgs).to(self.device)
                neg_batch = torch.stack(neg_imgs).to(self.device)
                
                anchor_emb = self.model(anchor_batch)
                pos_emb = self.model(pos_batch)
                neg_emb = self.model(neg_batch)
                
                loss = self.contrastive_loss_with_variance(anchor_emb, pos_emb, neg_emb)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"[CONTRAIN] Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")

            hardmining_needed = (epoch - warmup_epochs + 1) % hard_mine_interval == 0
            silhouette_eval_needed = (epoch - warmup_epochs + 1) % sil_check_interval == 0

            if (hardmining_needed or silhouette_eval_needed) and epoch < epochs - 1:
                self.model.eval()
                embeddings = self.extract_embeddings(dataset)

                if silhouette_eval_needed:
                    with torch.no_grad():
                        if len(embeddings) > sil_subsampling:
                            indices = np.random.choice(len(embeddings), sil_subsampling, replace=False)
                            sample_embs = embeddings[indices]
                        else:
                            sample_embs = embeddings

                        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(sample_embs)
                        score = silhouette_score(sample_embs, labels)
                        logger.info(f"[SILHOUETTE] Epoch {epoch+1}: Score = {score:.3f}")
        
                        if score > sil_threshold:
                            logger.info(f"[SILHOUETTE] Threshold {sil_threshold} reached. Stopping training early.")
                            break

                if hardmining_needed:   
                    logger.info(f"[CONTRAIN] Updating hard triplets at epoch {epoch+1}...")
                    hard_triplets = self.mine_hard_triplets_adaptive(dataset, embeddings,
                                                                    pos_threshold=pos_threshold,
                                                                    neg_threshold=neg_threshold,
                                                                    max_triplets=5000)
                    logger.info(f"[CONTRAIN] Re-mined {len(hard_triplets)} hard triplets")

                self.model.train()
        
        self.model.eval()
        return self.model

    def extract_embeddings(self, dataset):
        embeddings = []
        with torch.no_grad():
            for i in range(len(dataset)):
                crop, _, _, _ = dataset[i]
                tensor = self.transform(crop).unsqueeze(0).to(self.device)
                emb = self.model(tensor).squeeze().cpu().numpy()
                embeddings.append(emb)
        return np.vstack(embeddings)