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

from utils.dataclass import Emb_Params
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

        self.emb_pointer = 0

    def train(
            self,
            dataset:Crop_Dataset,
            emp:Emb_Params,
            margin_thresh = 0.5,
            sil_thresh = 0.5,
            ):
    
        logger.info(f"[CONTRAIN] About to run training with following args:\n"
            f"  - lr:                   {emp.lr}\n"
            f"  - epochs:               {emp.epochs}\n"
            f"  - warmup_epoch:         {emp.warmup}\n"
            f"  - batch_size:           {emp.batch_size}\n"
            f"  - max_triplet:          {emp.triplets}\n"
            f"  - pleatau_patience:     {emp.pleatau}\n")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=emp.lr)
        dataset.transform = self.transform

        self._train_easy(dataset, optimizer, emp)

        logger.info("[CONTRAIN] Extracting embeddings for eval...")
        embeddings = self.extract_embeddings(dataset)
        pos_thresh, neg_thresh = self._similarity_eval(dataset, embeddings, self.slide_window, emp.triplets)
        score = self._silhouette_eval(embeddings, emp.triplets)

        if pos_thresh - neg_thresh >= margin_thresh and score >= sil_thresh:
            logger.info(
                f"[CONTRAIN] Separation score: {pos_thresh - neg_thresh} >= {margin_thresh}; sihouette score: {score} >= {sil_thresh}. "
                "Stopping training early...")

            self.model.eval()
            return self.model
        
        self._train_hard(dataset, optimizer, emp, embeddings, pos_thresh, neg_thresh, margin_thresh, sil_thresh)
        
        self.model.eval()
        return self.model

    def extract_embeddings(self, dataset:Crop_Dataset, subsample_window=None, batch_size:int=128):
        embeddings = []
        self.model.eval()

        if subsample_window:
            start = max(0, subsample_window[0])
            end = min(len(dataset), subsample_window[1])
            window_indices = range(start, end)
        else:
            window_indices = range(len(dataset))

        with torch.no_grad():
            for i in range(0, len(window_indices), batch_size):
                batch_idx = window_indices[i:i+batch_size]
                batch_imgs = []
                for idx in batch_idx:
                    crop, _, _, _ = dataset[idx]
                    batch_imgs.append(crop)
                
                batch_tensors = torch.stack(batch_imgs).to(self.device)
                embs = self.model(batch_tensors).cpu().numpy()
                embeddings.append(embs)
        
        self.model.train()

        return np.vstack(embeddings)

    def _train_easy(self, dataset, optimizer:torch.optim.Adam, emp:Emb_Params):
        easy_triplets = self._mine_easy_triplets(dataset, window=5, max_triplets=emp.triplets)
        logger.info(f"[CONTRAIN] Mined {len(easy_triplets)} easy triplets")
        
        self.model.train()

        best_loss = 1e6
        pleatau_count = 0
        for epoch in range(emp.warmup):
            total_loss = 0
            np.random.shuffle(easy_triplets)
            num_batches = max(1, len(easy_triplets) // emp.batch_size)
            for batch_idx in range(num_batches):
                start = batch_idx * emp.batch_size
                end = start + emp.batch_size
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
                
                loss = self._contrastive_loss(anchor_emb, pos_emb, neg_emb, w_var=0.1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            curr_loss = total_loss/num_batches
            logger.info(f"[EASYTRAIN] Epoch {epoch+1}/{emp.warmup}, Loss: {curr_loss:.4e}")

            if curr_loss >= best_loss:
                pleatau_count += 1
            else:
                pleatau_count = 0

            if pleatau_count >= emp.pleatau:
                logger.info(f"[EASYTRAIN] Pleatau hit, switching to hard mining.")
                break
            
            best_loss = min(curr_loss, best_loss)

    def _train_hard(self, dataset, optimizer:torch.optim.Adam, emp:Emb_Params, embeddings, pos_thresh, neg_thresh, margin_thresh, sil_thresh):
        init_end = min(len(embeddings), emp.triplets*4)
        hard_triplets = self._mine_hard_triplets(dataset, embeddings[0:init_end], (0, init_end), pos_thresh, neg_thresh, max_triplets=emp.triplets, mining_mode="random")
        hardmine_interval = max(5, emp.warmup)
        eval_interval = max(10, emp.epochs//10)

        logger.info(f"[CONTRAIN] Training on hard triplets... Hardmine interval: {hardmine_interval}; Eval interval: {eval_interval}")
        self.model.train()

        curr_miner = "semihard"

        best_loss = 1e6
        pleatau_count = 0
        last_eval = emp.warmup - 1
        for epoch in range(emp.warmup, emp.epochs):
            total_loss = 0
            np.random.shuffle(hard_triplets)

            num_batches = max(1, len(hard_triplets) // emp.batch_size)
            for batch_idx in range(num_batches):
                start = batch_idx * emp.batch_size
                end = start + emp.batch_size
                batch_triplets = hard_triplets[start:end]
                
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
                
                loss = self._contrastive_loss(anchor_emb, pos_emb, neg_emb)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            curr_loss = total_loss/num_batches
            logger.info(f"[HARDTRAIN] Epoch {epoch+1}/{emp.epochs}, Loss: {curr_loss:.4e}")

            if curr_loss >= best_loss:
                pleatau_count += 1
            else:
                pleatau_count = 0

            force_eval = False
            force_hardmine = False
            if pleatau_count >= emp.pleatau:
                if curr_miner != "hard":
                    logger.info(f"[HARDTRAIN] Pleatau hit, switching to HARDER mining to finetune weights.")
                    curr_miner = "hard"
                    force_hardmine = True
                    pleatau_count = 0
                else:
                    logger.info(f"[HARDTRAIN] Pleatau hit with hardest mining, init global evaluation...")
                    force_eval = True
                    force_hardmine = True
            
            best_loss = min(curr_loss, best_loss)

            hardmining_needed = (epoch - emp.warmup + 1) % hardmine_interval == 0
            eval_needed = (epoch - last_eval) > eval_interval

            if eval_needed or force_eval or epoch == emp.epochs - 1:
                full_embeddings = self.extract_embeddings(dataset)
                pos_thresh, neg_thresh = self._similarity_eval(dataset, full_embeddings, self.slide_window, subsample_size=emp.triplets)
                score = self._silhouette_eval(full_embeddings, emp.triplets)

                last_eval = epoch
                if pos_thresh - neg_thresh >= margin_thresh and score > sil_thresh:
                    logger.info(
                        f"[HARDTRAIN] Separation score: {pos_thresh - neg_thresh} >= {margin_thresh}; sihouette score: {score} >= {sil_thresh}. "
                        "Stopping training early...")
                    break
                elif force_eval or epoch == emp.epochs - 1:
                    logger.info("[HARDTRAIN] The model did the best it could. Stopping early...")
                    break

            if hardmining_needed or force_hardmine:
                end = self.emb_pointer + emp.triplets
                subsample_window = (self.emb_pointer, end)
                local_embeddings = self.extract_embeddings(dataset, subsample_window)
                logger.info(f"[HARDTRAIN] Updating hard triplets at epoch {epoch+1}...")
                hard_triplets = self._mine_hard_triplets(
                    dataset, local_embeddings, subsample_window, pos_thresh, neg_thresh, max_triplets=emp.triplets, mining_mode=curr_miner)
                
                if not hard_triplets:
                    full_embeddings = self.extract_embeddings(dataset)
                    pos_thresh, neg_thresh = self._similarity_eval(dataset, full_embeddings, self.slide_window, subsample_size=emp.triplets)
                    score = self._silhouette_eval(full_embeddings, emp.triplets)

                    last_eval = epoch
                    if pos_thresh - neg_thresh >= margin_thresh and score > sil_thresh:
                        logger.info(
                            f"[HARDTRAIN] Separation score: {pos_thresh - neg_thresh} >= {margin_thresh}; sihouette score: {score} >= {sil_thresh}. "
                            "Stopping training early...")
                        break
                    else:
                        hard_triplets = self._mine_hard_triplets(
                            dataset, full_embeddings[subsample_window[0]:subsample_window[1]], subsample_window, pos_thresh, neg_thresh, max_triplets=emp.triplets, mining_mode=curr_miner)

                logger.info(f"[HARDTRAIN] Re-mined {len(hard_triplets)} hard triplets")
                self.emb_pointer = end
                if self.emb_pointer >= len(dataset):
                    self.emb_pointer = 0
                    logger.info("[HARDTRAIN] Completed full dataset sweep, cycling windows")

    def _mine_easy_triplets(self, dataset:Crop_Dataset, window:int=5, max_triplets:int=5000, subsample_size=None):
        if not subsample_size:
            subsample_size = max_triplets

        frames = sorted(dataset.frame_to_indices.keys())
        frames = frames[:subsample_size]

        triplets = []

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

    def _mine_hard_triplets(self, dataset:Crop_Dataset, embeddings, subsample_window, pos_threshold, neg_threshold, max_triplets=5000, mining_mode="semihard"):
        """
        mining_mode options:
            - "hard": Original (min pos, max neg)
            - "median": Selects ~50th percentile candidates
            - "semihard": Selects candidates near threshold boundaries
            - "random": Random selection from valid candidates
        """
        subsample_start, subsample_end = subsample_window
        motion_ids = np.array(dataset.motion_ids[subsample_start:subsample_end])
        frame_indices = np.array(dataset.frame_indices[subsample_start:subsample_end])
        sim_matrix = cosine_similarity(embeddings)

        triplets = []

        for i in range(len(embeddings)):
            anchor_id = motion_ids[i]
            frame_idx = frame_indices[i]

            same_mouse_mask = motion_ids == anchor_id
            same_mouse_sims = sim_matrix[i].copy()
            same_mouse_sims[~same_mouse_mask] = 2.0
            same_mouse_sims[i] = 2.0

            distant_mask = np.abs(frame_indices - frame_idx) >= self.slide_window
            same_mouse_sims[distant_mask] = 2.0

            if not np.any(same_mouse_mask):
                continue

            same_mouse_candidates = np.where(same_mouse_sims < pos_threshold)[0]
            if len(same_mouse_candidates) == 0:
                continue

            diff_mouse_mask = ~same_mouse_mask
            diff_mouse_sims = sim_matrix[i].copy()
            diff_mouse_sims[~diff_mouse_mask] = -2.0
            diff_mouse_sims[distant_mask] = -2.0

            if not np.any(diff_mouse_mask):
                continue

            diff_mouse_candidates = np.where(diff_mouse_sims > neg_threshold)[0]

            if len(diff_mouse_candidates) > 0:
                sim_values = same_mouse_sims[same_mouse_candidates]
                match mining_mode:
                    case "hard":
                        pos_idx = same_mouse_candidates[np.argmin(same_mouse_sims[same_mouse_candidates])]
                    case "median":
                        mid_idx = len(sim_values) // 2
                        pos_idx = same_mouse_candidates[mid_idx]
                    case "semihard":
                        target_pos = (pos_threshold + sim_values.min()) / 2
                        pos_diff = np.abs(sim_values - target_pos)
                        pos_idx = same_mouse_candidates[pos_diff.argmin()]
                    case "random":
                        pos_idx = np.random.choice(same_mouse_candidates)

                neg_sim_values = diff_mouse_sims[diff_mouse_candidates]
                match mining_mode:
                    case "hard":
                        neg_idx = diff_mouse_candidates[np.argmax(neg_sim_values)]
                    case "median":
                        mid_idx = len(diff_mouse_candidates) // 2
                        neg_idx = diff_mouse_candidates[mid_idx]
                    case "semihard":
                        target_neg = (neg_threshold + neg_sim_values.max()) / 2
                        neg_diff = np.abs(neg_sim_values - target_neg)
                        neg_idx = diff_mouse_candidates[neg_diff.argmin()]
                    case "random":
                        neg_idx = np.random.choice(diff_mouse_candidates)
                
                triplets.append((i+subsample_start, pos_idx+subsample_start, neg_idx+subsample_start))
            
            if len(triplets) >= max_triplets:
                break
        
        logger.info(f"[CONTRAIN] Found {len(triplets)} triplets (mode={mining_mode}, pos<{pos_threshold:.2f}, neg>{neg_threshold:.2f})")
        return triplets

    def _contrastive_loss(
            self,
            anchor_emb, pos_emb, neg_emb,
            margin:float=0.5,
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
    def _similarity_eval(dataset, embeddings, slide_window, subsample_size=5000):
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
        
        if not pos_sims or not neg_sims:
            raise ValueError("[CONTRAIN] Simularity score is empty!")

        logger.info(f"[CONTRAIN] Same-mouse: {np.mean(pos_sims):.3f} ± {np.std(pos_sims):.3f}")
        logger.info(f"[CONTRAIN] Diff-mouse: {np.mean(neg_sims):.3f} ± {np.std(neg_sims):.3f}")
        
        return np.percentile(pos_sims, 30), np.percentile(neg_sims, 70)

    @staticmethod
    def _silhouette_eval(embeddings, sil_subsampling:int=5000):
        total_len = len(embeddings)
        silhouette_scores = []

        n_chunks = max(1, (total_len + sil_subsampling - 1) // sil_subsampling)
        
        for i in range(n_chunks):
            start = i * sil_subsampling
            end = min((i + 1) * sil_subsampling, total_len)
            chunk_embs = embeddings[start:end]

            if len(chunk_embs) < 10:
                continue
                
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(chunk_embs)
                score = silhouette_score(chunk_embs, labels)
                silhouette_scores.append(score)
            except ValueError:
                continue

        if silhouette_scores:
            mean_score = np.mean(silhouette_scores)
            logger.info(f"[CONTRAIN] Silhouette score = {mean_score:.3f}")
            return mean_score
        else:
            return 0.0