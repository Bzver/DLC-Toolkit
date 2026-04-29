import os
import datetime
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from typing import List, Tuple, Dict

from .dataloader import Crop_Dataset
from core.io import backup_existing_prediction
from utils.dataclass import Emb_Params
from utils.logger import logger


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
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = Identity_Encoder().to(device)
        self.scaler = GradScaler(device=device) if device == 'cuda' else None
        self.use_amp = device == 'cuda' and torch.cuda.is_available()

    def train(
        self, 
        datasets: List[Crop_Dataset],
        emp: Emb_Params,
        pretrained: bool = False,
        ):

        n_segments = len(datasets) if isinstance(datasets, list) else 1
        total_frames = sum(len(ds) for ds in datasets) if isinstance(datasets, list) else len(dataset)     

        logger.info(f"[CONTRAIN] About to run training with following args:\n"
            f"  - lr:                   {emp.lr:.2e}\n"
            f"  - epochs:               {emp.epochs}\n"
            f"  - batch_size:           {emp.batch_size}\n"
            f"  - max_triplet:          {emp.triplets}\n"
            f"  - pleatau_patience:     {emp.pleatau}\n"
            f"  - margin_thresh:        {emp.margin:.2f}\n"
            f"  - min_improv:           {emp.min_imp:.2f}\n"
            f"  - n_segments:           {n_segments}\n"
            f"  - total_samples:        {total_frames}\n"
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=emp.lr)
        self.lr_reduced = False

        self.total_ds = len(datasets)
        reserve_indices = []
        if self.total_ds > 250 and not pretrained:
            reserve_indices = np.random.choice(self.total_ds, self.total_ds-250, replace=False).tolist()

        self.ds_status = np.ones(self.total_ds, dtype=np.uint8) # 0: unseen, 1: active, 2: good, 3: dropped
        self.ds_status[reserve_indices] = 0

        embeddings = self._extract_embeddings_list(datasets)
        sim_array, margin_array, weighted_mean = self._similarity_eval(embeddings, datasets)

        if pretrained:
            if self._check_early_stopping(emp, embeddings, weighted_mean):
                self.model.eval()
                return self.model

            self.ds_status[margin_array >= emp.margin] = 2
            self.ds_status[margin_array < emp.margin] = 3
            len_good = np.sum(self.ds_status==2)
            len_dropped = np.sum(self.ds_status==3)
            logger.info(f"[PREINIT] Eval done, moving {len_good} segments to good set, {len_dropped} segments to drop set.")

            embeddings = self._extract_embeddings_list(datasets)
            sim_array, _, _ = self._similarity_eval(embeddings, datasets)

        self._train_hard(datasets, optimizer, emp, embeddings, sim_array, weighted_mean)
        
        self.model.eval()
        return self.model

    def extract_embeddings(
        self,
        dataset: Crop_Dataset, 
        batch_size: int = 128,
    ) -> np.ndarray:

        embeddings = []
        self.model.eval()

        with torch.inference_mode():
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
            embs = []
            for batch_data in loader:
                if isinstance(batch_data, (list, tuple)):
                    batch_tensors = batch_data[0]
                else:
                    batch_tensors = batch_data

                batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                batch_embs = self.model(batch_tensors)
                embs.append(batch_embs)

            embeddings = torch.cat(embs, dim=0).cpu().numpy()
        
        self.model.train()
        return embeddings

    def save_checkpoint(self, path: str, metadata: dict = None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metadata': metadata or {},
            'timestamp': datetime.datetime.now().isoformat(),
            'torch_version': str(torch.__version__),
            'cuda_available': torch.cuda.is_available()
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        backup_existing_prediction(path)
        
        torch.save(checkpoint, path)
        logger.info(f"[TRAINER] Model saved to {path}")
        
        if metadata:
            logger.info(f"[TRAINER] Metadata: {metadata}")

    def load_checkpoint(self, path: str, strict: bool = False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logger.info(f"[TRAINER] Model loaded from {path}")
        except RuntimeError as e:
            if strict:
                raise
            else:
                logger.warning(f"[TRAINER] Partial load (strict=False): {e}")
                model_dict = self.model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
    
                matched_dict = {k: v for k, v in pretrained_dict.items() 
                               if k in model_dict and model_dict[k].shape == v.shape}
    
                model_dict.update(matched_dict)
                self.model.load_state_dict(model_dict, strict=False)
                logger.info(f"[TRAINER] Loaded {len(matched_dict)}/{len(pretrained_dict)} layers")

        metadata = checkpoint.get('metadata', {})
        if metadata:
            logger.info(f"[TRAINER] Checkpoint metadata: {metadata}")

        saved_torch_version = checkpoint.get('torch_version', 'unknown')
        if saved_torch_version != torch.__version__:
            logger.warning(f"[TRAINER] Torch version mismatch: saved={saved_torch_version}, current={torch.__version__}")
        
        return metadata

    def _extract_embeddings_list(
        self,
        datasets: List[Crop_Dataset],
        batch_size: int = 128,
    ) -> Dict[int, np.ndarray]:

        segment_embeddings = {}
        self.model.eval()

        n_active = min(250, len(datasets))

        len_dropped = np.sum(self.ds_status == 3)
        len_good = np.sum(self.ds_status == 2)
        len_active = np.sum(self.ds_status == 1)
        len_unseen = np.sum(self.ds_status == 0)

        if len_active < n_active and len_good > 0:
            good_indices = np.where(self.ds_status == 2)[0]
            good_needed = min(len_good, n_active//5)
            indices_to_add = good_indices[np.random.choice(len_good, good_needed, replace=False)]
            self.ds_status[indices_to_add] = 1

        len_active = np.sum(self.ds_status == 1)

        if len_active < n_active and len_unseen > 0:
            unseen_indices = np.where(self.ds_status == 0)[0]
            unseen_needed = min(len_unseen, n_active - len_active)
            indices_to_add = unseen_indices[np.random.choice(len_unseen, unseen_needed, replace=False)]
            self.ds_status[indices_to_add] = 1

        len_active = np.sum(self.ds_status == 1)

        if len_active < n_active and len_dropped > 0:
            dropped_indices = np.where(self.ds_status == 3)[0]
            dropped_needed = min(len_dropped, n_active - len_active)
            indices_to_add = dropped_indices[np.random.choice(len_dropped, dropped_needed, replace=False)]
            self.ds_status[indices_to_add] = 1

        with torch.inference_mode():
            for ds_idx, ds in enumerate(datasets):
                if self.ds_status[ds_idx] != 1:
                    segment_embeddings[ds_idx] = np.full(1, np.nan)
                    continue

                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

                embs = []
                for batch_data in loader:
                    if isinstance(batch_data, (list, tuple)):
                        batch_tensors = batch_data[0]
                    else:
                        batch_tensors = batch_data
    
                    batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                    batch_embs = self.model(batch_tensors)
                    embs.append(batch_embs)

                if embs:
                    segment_embeddings[ds_idx] = torch.cat(embs, dim=0).cpu().numpy()
                
        self.model.train()
        return segment_embeddings

    def _train_hard(
            self, 
            datasets: List[Crop_Dataset],
            optimizer: torch.optim.Adam, 
            emp: Emb_Params, 
            embeddings: Dict[int, np.ndarray],
            sim_array: np.ndarray,
            mean_margin: float
    ):

        first_mine_mode = "random" if mean_margin < 0.1 else "semihard"

        hard_triplets = self._mine_hard_triplets(
            datasets, embeddings, sim_array[:, 1, 0], sim_array[:, 2, 1], max_triplets=emp.triplets, mining_mode=first_mine_mode
        )

        if len(hard_triplets) == 0:
            logger.warning("[HARDTRAIN] No hard triplets found. Model may already be well-separated.")
            logger.warning("[HARDTRAIN] Skipping hard training phase, proceeding with current model state.")
            return
        
        eval_interval = 5
        for pleatau_tier in [5, 10, 25, 50, 75, 100]:
            if emp.pleatau < pleatau_tier:
                eval_interval = pleatau_tier
                break

        logger.info(f"[CONTRAIN] Training on hard triplets with eval interval set to {eval_interval}...")
        self.model.train()

        best_loss = 1e6
        pleatau_train = 0

        last_eval = - 1
        best_eval = 0.0
        pleatau_eval = 2

        for epoch in range(emp.epochs):
            total_loss = 0
            np.random.shuffle(hard_triplets)

            num_batches = max(1, len(hard_triplets) // emp.batch_size)

            if num_batches == 0:
                logger.warning(f"[HARDTRAIN] Epoch {epoch+1}: No batches available (only {len(hard_triplets)} triplets)")
                break
            
            for batch_idx in range(num_batches):
                start = batch_idx * emp.batch_size
                end = start + emp.batch_size
                batch_triplets = hard_triplets[start:end]
                
                anchor_imgs = []
                pos_imgs = []
                neg_imgs = []
                
                for ds_idx, anchor_idx, pos_idx, neg_idx in batch_triplets:
                    ds = datasets[ds_idx]
                    anchor_img, _, _, _ = ds[anchor_idx]
                    pos_img, _, _, _ = ds[pos_idx]
                    neg_img, _, _, _ = ds[neg_idx]
                    
                    anchor_imgs.append(anchor_img)
                    pos_imgs.append(pos_img)
                    neg_imgs.append(neg_img)

                if len(anchor_imgs) == 0:
                    continue
                
                anchor_batch = torch.stack(anchor_imgs).to(self.device)
                pos_batch = torch.stack(pos_imgs).to(self.device)
                neg_batch = torch.stack(neg_imgs).to(self.device)
                
                with autocast(self.device):
                    anchor_emb = self.model(anchor_batch)
                    pos_emb = self.model(pos_batch)
                    neg_emb = self.model(neg_batch)
                    loss = self._contrastive_loss(anchor_emb, pos_emb, neg_emb, margin=emp.margin, w_compat=0.1)

                optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()

            curr_loss = total_loss/num_batches if num_batches > 0 else 0
            logger.info(f"[HARDTRAIN] Epoch {epoch+1}/{emp.epochs}, Loss: {curr_loss:.4e}")

            if curr_loss >= best_loss and curr_loss > 0:
                pleatau_train += 1
                logger.info(f"[HARDTRAIN] Current training pleatau: {pleatau_train}, best loss: {best_loss:.4e}, patience: {emp.pleatau}")
            else:
                pleatau_train = 0

            if pleatau_train >= emp.pleatau:
                logger.info(f"[HARDTRAIN] Loss plateaued for {emp.pleatau} epochs. Calling eval...")
                eval_needed = True
                best_loss = 1e6
            else:       
                eval_needed = (epoch - last_eval) % eval_interval == 0
            
            best_loss = min(curr_loss, best_loss) if curr_loss > 0 else best_loss

            if eval_needed or epoch == emp.epochs - 1:
                last_eval = epoch
                pleatau_train = 0

                embeddings = self._extract_embeddings_list(datasets)
                _, margin_array, weighted_mean = self._similarity_eval(embeddings, datasets)

                self.ds_status[margin_array >= emp.margin] = 2
                self.ds_status[margin_array < emp.margin] = 3
                len_good = np.sum(self.ds_status==2)
                len_dropped = np.sum(self.ds_status==3)
                logger.info(f"[HARDTRAIN] Eval done, all good segments: {len_good}, all dropped segments: {len_dropped}.")

                eval_score = weighted_mean

                len_unseen = np.sum(self.ds_status==0)
                coverage_pct = (1 - len_unseen / len(datasets)) * 100
    
                if eval_score < best_eval and np.all(self.ds_status != 0):
                    logger.info(f"[HARDTRAIN] Eval score ({eval_score}) < best ({best_eval}) while all data has been itered. pleatau: {pleatau_eval}")
                    pleatau_eval += 1
                else:
                    pleatau_eval = 0

                if len_good > len(datasets) * 0.9:
                    logger.info("[HARDTRAIN] Most segments are good. Stopping early.")
                    break
                else:
                    logger.info(f"[HARDTRAIN] {coverage_pct:.1f}% of segments evaluated at least once "
                        f"(Unseen remaining: {len_unseen}/{len(datasets)})")
                if pleatau_eval > 1:
                    logger.info(f"[HARDTRAIN] Eval score pleataued, calling global eval.")
                    self.ds_status[:] = 1
                    embeddings = self._extract_embeddings_list(datasets)
                    _, margin_array, weighted_mean = self._similarity_eval(embeddings, datasets)

                    if self._check_early_stopping(emp, embeddings, weighted_mean):
                        logger.info("[HARDTRAIN] Thresholds met on convergence. Stopping early.")
                    else:
                        logger.info("[HARDTRAIN] Model did the best it could. Stopping early.")
                    break
                if self._check_early_stopping(emp, embeddings, weighted_mean):
                    if not self.lr_reduced:
                        old_lr = optimizer.param_groups[0]['lr']
                        new_lr = old_lr * 0.1 
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        self.lr_reduced = True
                        logger.info(f"[HARDTRAIN] Threshold met, dropping LR: {old_lr:.2e} → {new_lr:.2e}")

                    len_unseen = np.sum(self.ds_status==0)
                    len_dropped = np.sum(self.ds_status==3)
                    if len_unseen + len_dropped > self.total_ds // 5:
                        logger.info(f"[CONTRAIN] Unseen/dropped datasets ({len_unseen + len_dropped}) too many (>{self.total_ds//4})")
                    else:
                        logger.info("[HARDTRAIN] Thresholds met on convergence. Stopping early.")
                        break
                if epoch == emp.epochs - 1:
                    logger.info("[HARDTRAIN] Max epoch reached. Stopping...")
                    break

                logger.info(f"[HARDTRAIN] Updating hard triplets at epoch {epoch+1}.")

                embeddings = self._extract_embeddings_list(datasets)
                sim_array, _, _ = self._similarity_eval(embeddings, datasets)

                result = self._mine_hard_triplets(
                    datasets, embeddings, sim_array[:, 1, 0], sim_array[:, 2, 1], max_triplets=emp.triplets)
                num_result = len(result)
                if num_result < 10:
                    logger.info(f"[HARDTRAIN] Hard triplets too few ({num_result}, stopping...")
                    break

                hard_triplets = result
                logger.info(f"[HARDTRAIN] Re-mined {num_result} hard triplets")
                
                best_eval = max(eval_score, best_eval)

    def _mine_hard_triplets(
            self,
            datasets: List[Crop_Dataset],
            segment_embeddings: Dict[int, np.ndarray],
            pos_thresholds: np.ndarray,
            neg_thresholds: np.ndarray,
            max_triplets: int = 5000,
            mining_mode: str = "semihard"
    ) -> List[Tuple[int, int, int, int]]:

        triplets = []
        triplet_quota = (max_triplets + 1) // len(datasets) + 1

        for ds_idx, dataset in enumerate(datasets):
            if len(triplets) >= max_triplets:
                break

            triplets_ds = []

            if ds_idx not in segment_embeddings:
                continue
            if self.ds_status[ds_idx] != 1:
                continue
    
            pos_threshold = pos_thresholds[ds_idx]
            neg_threshold = neg_thresholds[ds_idx]
            
            embeddings = segment_embeddings[ds_idx]
            motion_ids = np.array(dataset.motion_ids[:len(embeddings)])
            
            if len(embeddings) < 3:
                continue
            
            sim_matrix = cosine_similarity(embeddings)

            for i in range(len(embeddings)):
                if len(triplets_ds) >= triplet_quota:
                    break

                anchor_id = motion_ids[i]

                same_mouse_mask = motion_ids == anchor_id
                same_mouse_sims = sim_matrix[i].copy()
                same_mouse_sims[~same_mouse_mask] = 2.0
                same_mouse_sims[i] = 2.0

                if not np.any(same_mouse_mask):
                    continue

                same_mouse_candidates = np.where(same_mouse_sims < pos_threshold)[0]
                if len(same_mouse_candidates) == 0:
                    continue

                diff_mouse_sims = sim_matrix[i].copy()
                diff_mouse_sims[same_mouse_mask] = -2.0

                diff_mouse_candidates = np.where(diff_mouse_sims > neg_threshold)[0]
                if len(diff_mouse_candidates) == 0:
                    continue

                sim_values = same_mouse_sims[same_mouse_candidates]
                match mining_mode:
                    case "hard":
                        pos_idx = same_mouse_candidates[np.argmin(sim_values)]
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
                
                triplets_ds.append((ds_idx, i, pos_idx, neg_idx))

            triplets.extend(triplets_ds)

        logger.info(f"[CONTRAIN] Found {len(triplets)} hard triplets (mode={mining_mode}, avg_pos_thresh={np.nanmean(pos_thresholds):.2f}, avg_neg_thresh={np.nanmean(neg_thresholds):.2f})")
        return triplets

    def _check_early_stopping(self, emp, embeddings, weighted_mean):
        logger.info(f"[CONTRAIN] Separation score: {weighted_mean}, margin: {emp.margin}" )
        if weighted_mean > emp.margin:
            logger.info(f"[CONTRAIN] Separation score satisfied, calling biomodal evaluation." )
            if self._biomodal_eval(embeddings):
                return True
            
        return False

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
    def _similarity_eval(embeddings:Dict[int, np.ndarray], datasets) -> Tuple[np.ndarray, np.ndarray, float]:
        sim_array = np.full((len(embeddings), 3, 2), np.nan)
        dslen_array = np.full(len(embeddings), np.nan)

        for ds_idx, emb in embeddings.items():
            if emb.size == 1:
                continue

            dataset = datasets[ds_idx]
            sim_sub = cosine_similarity(emb)
            n_sub = len(emb)
            
            motion_ids = np.array(dataset.motion_ids[:len(emb)])
            
            i_idx, j_idx = np.triu_indices(n_sub, k=1)
            valid_sims = sim_sub[i_idx, j_idx]
            same_mouse = motion_ids[i_idx] == motion_ids[j_idx]
            
            pos_sim = valid_sims[same_mouse]
            neg_sim = valid_sims[~same_mouse]
    
            dslen_array[ds_idx] = len(valid_sims)
            for k, sim in enumerate([pos_sim, neg_sim]):
                sim_array[ds_idx, 0, k] = np.mean(sim)
                sim_array[ds_idx, 1, k] = np.percentile(sim, 30)
                sim_array[ds_idx, 2, k] = np.percentile(sim, 70)

        margin_array = sim_array[:, 0, 0] - sim_array[:, 0, 1]
        valid_mask = ~(np.isnan(margin_array) | np.isnan(dslen_array))
        weighted_mean = np.average(margin_array[valid_mask], weights=dslen_array[valid_mask])

        return sim_array, margin_array, weighted_mean

    @staticmethod
    def _biomodal_eval(embeddings: Dict[int, np.ndarray]) -> bool:
        valid_embs = [emb for emb in embeddings.values() if emb.size > 1]
        
        if not valid_embs:
            logger.warning("[BIOMODAL] No valid embeddings found for evaluation.")
            return False

        total_points = sum(emb.shape[0] for emb in valid_embs)
        if total_points < 100:
            logger.warning(f"[BIOMODAL] Too few points ({total_points}) for reliable bimodality test.")
            return False

        all_embeddings = np.vstack([emb for emb in valid_embs]).astype(np.float32)
        
        kmeans = MiniBatchKMeans(
            n_clusters=2,
            random_state=42,
            batch_size=min(1024, total_points),
            n_init=10,
            max_iter=300,
            reassignment_ratio=0.01,
            verbose=0
        )
        
        kmeans.fit_predict(all_embeddings)
        centroids = kmeans.cluster_centers_

        data_norms_sq = np.sum(all_embeddings**2, axis=1)
        centroid_norms_sq = np.sum(centroids**2, axis=1)
        
        dot_products = np.dot(all_embeddings, centroids.T)
        dists_sq = data_norms_sq[:, np.newaxis] + centroid_norms_sq - 2 * dot_products
        
        dists_sq = np.maximum(dists_sq, 0.0)
        distances_to_centroids = np.sqrt(dists_sq) 
        
        dist_to_c0 = distances_to_centroids[:, 0]
        dist_to_c1 = distances_to_centroids[:, 1]

        assign_margin = np.abs(dist_to_c0 - dist_to_c1)

        all_distances = np.hstack([dist_to_c0, dist_to_c1])
        result = Contrastive_Trainer._compute_bimodality_score(all_distances)
        
        std_margin = np.std(assign_margin)
        confidence_cv = np.mean(assign_margin) / (std_margin + 1e-6)

        logger.info(
            f"[BIOMODAL] Score={result['score']:.3f}, Peak Ratio={result['peak_ratio']:.3f}, "
            f"Valley Depth={result['valley_depth']:.3f}, CV={confidence_cv:.3f}, N={total_points}"
        )

        if result['bimodal'] and result["score"] > 0.9 and result["center_deviation"] < 0.1 and result["valley_depth"] > 0.8 and confidence_cv > 5:
            logger.info(f"[CROSSTRAIN] Clean bimodal separation achieved. Stopping early.")
            return True
        
        return False

    @staticmethod
    def _compute_bimodality_score(
            distances: np.ndarray, 
            min_valley_depth: float = 0.90,
            max_center_deviation: float = 0.15,
            min_peak_separation_frac: float = 0.15,
            kde_bw_factor: float = 0.2
            ) -> Dict[str, any]:
        
        default_result = {
            'bimodal': False,
            'score': 0.0,
            'valley_depth': 0.0,
            'peak_ratio': 0.0,
            'center_deviation': 1.0,
            'peak_positions': (0.0, 0.0),
            'valley_position': 0.0
        }

        if len(distances) < 50:
            return default_result

        x_data = distances.reshape(1, -1)
        kde = gaussian_kde(x_data, bw_method=kde_bw_factor)
        data_min, data_max = np.min(distances), np.max(distances)
        data_range = data_max - data_min
        
        x_grid = np.linspace(data_min - 0.1 * data_range, data_max + 0.1 * data_range, 1000)
        y_grid = kde(x_grid)

        global_max = np.max(y_grid)
        if global_max < 1e-9:
            return default_result
            
        peaks, properties = find_peaks(y_grid, prominence=0.25 * global_max, distance=50)
        
        if len(peaks) < 2:
            return default_result
        
        prominences = properties['prominences']
        top_two_indices = np.argsort(prominences)[-2:][::-1]
        
        p1_idx = peaks[top_two_indices[0]]
        p2_idx = peaks[top_two_indices[1]]
        
        if p1_idx > p2_idx:
            p1_idx, p2_idx = p2_idx, p1_idx
            
        p1_pos = x_grid[p1_idx]
        p2_pos = x_grid[p2_idx]
        h1 = y_grid[p1_idx]
        h2 = y_grid[p2_idx]

        if (p2_pos - p1_pos) < (data_range * min_peak_separation_frac):
            return default_result

        valley_region = y_grid[p1_idx : p2_idx + 1]
        valley_min_val = np.min(valley_region)
        valley_idx = p1_idx + np.argmin(valley_region)
        valley_pos = x_grid[valley_idx]
        
        mean_peak_height = (h1 + h2) / 2.0
        
        valley_depth = 1.0 - (valley_min_val / (mean_peak_height + 1e-6))
        
        expected_midpoint_pos = (p1_pos + p2_pos) / 2.0
        distance_between_peaks = p2_pos - p1_pos
        
        if distance_between_peaks == 0:
            return default_result
            
        center_deviation = abs(valley_pos - expected_midpoint_pos) / (distance_between_peaks / 2.0)

        peak_ratio = min(h1, h2) / (max(h1, h2) + 1e-6)
        
        if valley_depth < min_valley_depth:
            return {**default_result, 'valley_depth': float(valley_depth), 'center_deviation': float(center_deviation)}
            
        if center_deviation > max_center_deviation:
            return {**default_result, 'valley_depth': float(valley_depth), 'center_deviation': float(center_deviation)}

        centering_score = 1.0 - center_deviation
        final_score = (0.7 * valley_depth) + (0.3 * centering_score)
        final_score = np.clip(final_score, 0.0, 1.0)
        
        return {
            'bimodal': True,
            'score': float(final_score),
            'valley_depth': float(valley_depth),
            'peak_ratio': float(peak_ratio),
            'center_deviation': float(center_deviation),
            'peak_positions': (float(p1_pos), float(p2_pos)),
            'valley_position': float(valley_pos)
        }