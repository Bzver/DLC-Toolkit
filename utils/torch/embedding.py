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

        self.mined_frames = set()

    def train(
            self,
            dataset:Crop_Dataset,
            batch_size=64,
            lr=1e-5,
            epochs=20,
            max_triplet=5000,
            sil_threshold=0.4,
            diff_threshold=0.4,
            ):
    
        logger.info(f"[CONTRAIN] About to run training with following args:\n"
            f"  - lr:              {lr}\n"
            f"  - epochs:          {epochs}\n"
            f"  - batch_size:      {batch_size}\n"
            f"  - max_triplet:     {max_triplet}\n"
            f"  - sil_threshold:   {sil_threshold}\n"
            f"  - diff_threshold:  {diff_threshold}\n")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        dataset.transform = self.transform
        easy_triplets = self._mine_triplets(dataset, window=5, max_triplets=max_triplet)
        logger.info(f"[CONTRAIN] Mined {len(easy_triplets)} easy triplets")
        
        self.model.train()
        for epoch in range(epochs):
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
                    
                    anchor_imgs.append(anchor_img)
                    pos_imgs.append(pos_img)
                    neg_imgs.append(neg_img)
                
                anchor_batch = torch.stack(anchor_imgs).to(self.device)
                pos_batch = torch.stack(pos_imgs).to(self.device)
                neg_batch = torch.stack(neg_imgs).to(self.device)
                
                anchor_emb = self.model(anchor_batch)
                pos_emb = self.model(pos_batch)
                neg_emb = self.model(neg_batch)
                
                loss = self._contrastive_loss(anchor_emb, pos_emb, neg_emb, w_compat=0.1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                embeddings = self.extract_embeddings(dataset)
                
                same_mean, diif_mean = self._log_similarity_metrics(dataset, embeddings, self.slide_window, subsample_size=max_triplet)
                score = self._silhouette_eval(embeddings, sil_subsampling=max_triplet)
                logger.info(f"[SILHOUETTE] Epoch {epoch+1}: Score = {score:.3f}")
                
                if score > sil_threshold and (same_mean > 1 - diff_threshold) and (diif_mean < diff_threshold):
                    logger.info(f"[SILHOUETTE] Threshold {sil_threshold} reached. Stopping training early.")
                    break

                if epoch < epochs - 1:
                    logger.info(f"[CONTRAIN] Re-mining fresh triplets at epoch {epoch+1}...")
                    easy_triplets = self._mine_triplets(dataset, window=5, max_triplets=max_triplet)
                    logger.info(f"[CONTRAIN] Re-mined {len(easy_triplets)} new triplets")

            logger.info(f"[CONTRAIN] Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4e}")

        self.model.eval()
        return self.model

    def extract_embeddings(self, dataset:Crop_Dataset, batch_size:int=128):
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch_indices = range(i, min(i + batch_size, len(dataset)))
                
                anchor_imgs = []
                for idx in batch_indices:
                    crop, _, _, _ = dataset[idx]
                    anchor_imgs.append(crop)
                
                batch_tensors = torch.stack(anchor_imgs).to(self.device)
                embs = self.model(batch_tensors).cpu().numpy()
                embeddings.append(embs)
        
        return np.vstack(embeddings)

    def _mine_triplets(self, dataset:Crop_Dataset, window:int=5, max_triplets:int=5000):
        triplets = []
        frames = sorted(dataset.frame_to_indices.keys())

        for frame_idx in frames:
            indices_in_frame = dataset.frame_to_indices[frame_idx]
            if len(indices_in_frame) != 2:
                continue
            if frame_idx in self.mined_frames:
                continue
            
            idx_a, idx_b = indices_in_frame
            self.mined_frames.add(frame_idx)

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
        
        self.mined_frames.clear()
        return triplets

    def _contrastive_loss(
            self,
            anchor_emb, pos_emb, neg_emb,
            margin:float=0.75,
            w_compat:float=0.0,
            w_var:float=0.0,
            ):

        pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
        triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()
        compat_loss = 0 if w_compat < 1e-3 else self._compat_loss(pos_dist, w_compat)
        var_loss = 0 if w_var < 1e-3 else self._variance_loss(anchor_emb, pos_emb, neg_emb, w_var)

        return triplet_loss + compat_loss + var_loss

    @staticmethod
    def _compat_loss(pos_dist, compact_weight=0.1):
        compact_loss = pos_dist.mean()
        return compact_weight * compact_loss
    
    @staticmethod
    def _variance_loss(anchor_emb, pos_emb, neg_emb, var_weight=0.1):
        all_embs = torch.cat([anchor_emb, pos_emb, neg_emb], dim=0)
        var_loss = torch.mean(torch.var(all_embs, dim=0))
        var_penalty = torch.abs(var_loss - 1.0)
        return var_weight * var_penalty

    @staticmethod
    def _log_similarity_metrics(dataset, embeddings, slide_window, subsample_size=5000):
        motion_ids_arr = np.array(dataset.motion_ids)
        frame_indices_arr = np.array(dataset.frame_indices)
        total_len = len(embeddings)
        
        pos_sims, neg_sims = [], []

        n_chunks = max(1, (total_len + subsample_size - 1) // subsample_size)
        for i in range(n_chunks):
            start = i * subsample_size
            end = min((i + 1) * subsample_size, total_len)

            sub_embs = embeddings[start:end]
            sub_motion = motion_ids_arr[start:end]
            sub_frames = frame_indices_arr[start:end]

            if len(sub_embs) < 10:
                continue

            sim_sub = cosine_similarity(sub_embs)
            n_sub = len(sub_embs)

            i_idx, j_idx = np.triu_indices(n_sub, k=1)
            frame_diffs = np.abs(sub_frames[j_idx] - sub_frames[i_idx])
            valid_mask = frame_diffs <= slide_window

            valid_sims = sim_sub[i_idx[valid_mask], j_idx[valid_mask]]
            same_mouse = sub_motion[i_idx[valid_mask]] == sub_motion[j_idx[valid_mask]]

            pos_sims.extend(valid_sims[same_mouse].tolist())
            neg_sims.extend(valid_sims[~same_mouse].tolist())
        
        if pos_sims and neg_sims:
            logger.info(f"[CONTRAIN] Same-mouse: {np.mean(pos_sims):.3f} ± {np.std(pos_sims):.3f}")
            logger.info(f"[CONTRAIN] Diff-mouse: {np.mean(neg_sims):.3f} ± {np.std(neg_sims):.3f}")

        return np.mean(pos_sims), np.mean(neg_sims)
        
    @staticmethod
    def _silhouette_eval(embeddings, sil_subsampling:int=5000):
        if len(embeddings) > sil_subsampling:
            indices = np.random.choice(len(embeddings), sil_subsampling, replace=False)
            sample_embs = embeddings[indices]
        else:
            sample_embs = embeddings

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sample_embs)
        return silhouette_score(sample_embs, labels)