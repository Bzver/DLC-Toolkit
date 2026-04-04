import os
import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from typing import List, Tuple, Dict

from .dataloader import Crop_Dataset
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

    def train(
            self,
            emp: Emb_Params,
            dataset: List[Crop_Dataset], 
            skip_easy: bool = False,
            ):

        n_segments = len(dataset) if isinstance(dataset, list) else 1
        total_frames = sum(len(ds) for ds in dataset) if isinstance(dataset, list) else len(dataset)
        
        logger.info(f"[CONTRAIN] About to run training with following args:\n"
            f"  - lr:                   {emp.lr:.2e}\n"
            f"  - epochs:               {emp.epochs}\n"
            f"  - warmup_epoch:         {emp.warmup}\n"
            f"  - batch_size:           {emp.batch_size}\n"
            f"  - max_triplet:          {emp.triplets}\n"
            f"  - pleatau_patience:     {emp.pleatau}\n"
            f"  - margin_thresh:        {emp.margin:.2f}\n"
            f"  - sil_thresh:           {emp.sil:.2f}\n"
            f"  - min_improv:           {emp.min_imp:.2f}\n"
            f"  - n_segments:           {n_segments}\n"
            f"  - total_samples:        {total_frames}\n"
            )

        optimizer_warmup = torch.optim.Adam(self.model.parameters(), lr=emp.lr*10)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=emp.lr)

        if not skip_easy and emp.warmup != 0:
            self._train_easy(dataset, optimizer_warmup, emp)

        embeddings = self._extract_embeddings_list(dataset)

        pos_thresh, neg_thresh = self._similarity_eval(embeddings)
        score = self._silhouette_eval(embeddings)

        if pos_thresh - neg_thresh >= emp.margin and score >= emp.sil:
            logger.info(
                f"[CONTRAIN] Separation score: {pos_thresh - neg_thresh} >= {emp.margin}; sihouette score: {score} >= {emp.sil}. "
                "Stopping training early...")

            self.model.eval()
            return self.model

        self._train_hard(dataset, optimizer, emp, embeddings, pos_thresh, neg_thresh, skip_easy)
        
        self.model.eval()
        return self.model

    def extract_embeddings(
            self,
            dataset: Crop_Dataset, 
            batch_size: int = 128
    ) -> np.ndarray:

        embeddings = []
        self.model.eval()

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch_idx = range(i, min(i + batch_size, len(dataset)))
                batch_imgs = [dataset[idx][0] for idx in batch_idx]
                batch_tensors = torch.stack(batch_imgs).to(self.device)
                embs = self.model(batch_tensors).cpu().numpy()
                embeddings.append(embs)
        
        self.model.train()
        return np.vstack(embeddings) if embeddings else np.array([])

    def eval_on_dataset(
            self,
            val_dataset: List[Crop_Dataset],
            batch_size: int = 128,
    ) -> Tuple[float, float]:

        val_emb = self._extract_embeddings_list(val_dataset, batch_size)
        pos_thresh, neg_thresh = self._similarity_eval(val_emb)
        val_sil = self._silhouette_eval(val_emb)

        return pos_thresh, neg_thresh, val_sil

    def save_checkpoint(self, path: str, metadata: dict = None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metadata': metadata or {},
            'timestamp': datetime.datetime.now().isoformat(),
            'torch_version': str(torch.__version__),
            'cuda_available': torch.cuda.is_available()
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
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
        
        for ds_idx, ds in enumerate(datasets):
            embs = []
            with torch.no_grad():
                for i in range(0, len(ds), batch_size):
                    batch_idx = range(i, min(i + batch_size, len(ds)))
                    batch_imgs = [ds[idx][0] for idx in batch_idx]
                    batch_tensors = torch.stack(batch_imgs).to(self.device)
                    batch_embs = self.model(batch_tensors).cpu().numpy()
                    embs.append(batch_embs)
            
            if embs:
                segment_embeddings[ds_idx] = np.vstack(embs)
        
        self.model.train()
        return segment_embeddings

    def _train_easy(
            self, 
            datasets: List[Crop_Dataset],
            optimizer: torch.optim.Adam, 
            emp: Emb_Params
    ):

        easy_triplets = self._mine_easy_triplets(datasets, window=20, max_triplets=emp.triplets)
        logger.info(f"[EASYTRAIN] Mined {len(easy_triplets)} easy triplets from {len(datasets)} segments")
        
        self.model.train()

        best_loss = 1e6
        pleatau_train = 0
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
                
                for ds_idx, anchor_idx, pos_idx, neg_idx in batch_triplets:
                    ds = datasets[ds_idx]
                    anchor_img, _, _, _ = ds[anchor_idx]
                    pos_img, _, _, _ = ds[pos_idx]
                    neg_img, _, _, _ = ds[neg_idx]
                    
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

            if curr_loss >= best_loss or curr_loss == 0:
                pleatau_train += 1
            else:
                pleatau_train = 0

            if pleatau_train >= emp.pleatau:
                logger.info(f"[EASYTRAIN] Pleatau hit, switching to hard mining.")
                break
            
            best_loss = min(curr_loss, best_loss)

    def _train_hard(
            self, 
            datasets: List[Crop_Dataset],
            optimizer: torch.optim.Adam, 
            emp: Emb_Params, 
            embeddings: np.ndarray, 
            pos_thresh: float, 
            neg_thresh: float,
            skip_easy: bool = False
    ):

        hard_triplets = self._mine_hard_triplets(
            datasets, embeddings, pos_thresh, neg_thresh, max_triplets=emp.triplets, mining_mode="semihard"
        )

        if len(hard_triplets) == 0:
            logger.warning("[HARDTRAIN] No hard triplets found. Model may already be well-separated.")
            logger.warning("[HARDTRAIN] Skipping hard training phase, proceeding with current model state.")
            return
        
        eval_interval = max(5, emp.epochs//20)

        logger.info(f"[CONTRAIN] Training on hard triplets...")
        self.model.train()

        curr_miner = "hard" if skip_easy else "semihard"

        best_loss = 1e6
        pleatau_train = 0

        last_eval = emp.warmup - 1
        best_eval = 0.0
        pleatau_eval = 0

        segment_embeddings = self._extract_embeddings_list(datasets)

        for epoch in range(emp.warmup, emp.epochs):
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
                
                anchor_emb = self.model(anchor_batch)
                pos_emb = self.model(pos_batch)
                neg_emb = self.model(neg_batch)
                
                loss = self._contrastive_loss(anchor_emb, pos_emb, neg_emb, w_compat=0.1)
                
                optimizer.zero_grad()
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
                if curr_miner == "semihard":
                    curr_miner = "hard"
                    best_loss = float('inf')
                    pleatau_train = 0
                    logger.info("[HARDTRAIN] Switching to HARD mining. Resetting patience.")
                    continue
    
                logger.info(f"[HARDTRAIN] Loss plateaued for {emp.pleatau} epochs in hard mode. Evaluating final state...")

                pos_thresh, neg_thresh, sil = self.eval_on_dataset(datasets)
                if pos_thresh - neg_thresh >= emp.margin and sil >= emp.sil:
                    logger.info("[HARDTRAIN] Thresholds met on convergence. Stopping early.")
                else:
                    logger.info("[HARDTRAIN] Model converged to best possible state. Stopping early.")
                break 
            
            best_loss = min(curr_loss, best_loss) if curr_loss > 0 else best_loss
            eval_needed = (epoch - last_eval) % eval_interval == 0

            if eval_needed or epoch == emp.epochs - 1:
                last_eval = epoch

                pos_thresh, neg_thresh, sil = self.eval_on_dataset(datasets)
                eval_score = min(pos_thresh - neg_thresh, sil)

                eval_patience = max(3, emp.pleatau//4)
                if eval_score - best_eval < emp.min_imp:
                    pleatau_eval += 1
                    logger.info(f"[HARDTRAIN] Current eval pleatau: {pleatau_eval}, last improvement: {eval_score - best_eval:.3f}, patience: {eval_patience}")

                if pos_thresh - neg_thresh >= emp.margin and sil >= emp.sil:
                    logger.info("[HARDTRAIN] Thresholds met on convergence. Stopping early.")
                    break
                elif epoch == emp.epochs - 1:
                    logger.info("[HARDTRAIN] Max epoch reached. Stopping...")
                    break
                elif pleatau_eval >= eval_patience:
                    logger.info("[HARDTRAIN] Model converged to best possible state. Stopping early.")
                    break
                else:
                    segment_embeddings = self._extract_embeddings_list(datasets)
                    logger.info(f"[HARDTRAIN] Updating hard triplets at epoch {epoch+1}...")
                    result = self._mine_hard_triplets(datasets, segment_embeddings, pos_thresh, neg_thresh, max_triplets=emp.triplets, mining_mode=curr_miner)
                    num_result = len(result)
                    if num_result < 10:
                        logger.info(f"[HARDTRAIN] Hard triplets too few ({num_result}, stopping...")
                        break
                    else:
                        hard_triplets = result
                        logger.info(f"[HARDTRAIN] Re-mined {num_result} hard triplets")
                
                best_eval = max(eval_score, best_eval)

    def _mine_easy_triplets(
            self, 
            datasets: List[Crop_Dataset],
            window: int = 5, 
            max_triplets: int = 5000
    ) -> List[Tuple[int, int, int, int]]:

        triplets = []

        for ds_idx, dataset in enumerate(datasets):
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
                                triplets.append((ds_idx, idx_a, next_idx, idx_b))
                                if len(triplets) >= max_triplets:
                                    return triplets
                        break

        return triplets

    def _mine_hard_triplets(
            self,
            datasets: List[Crop_Dataset],
            segment_embeddings: Dict[int, np.ndarray],
            pos_threshold: float,
            neg_threshold: float,
            max_triplets: int = 5000,
            mining_mode: str = "semihard"
    ) -> List[Tuple[int, int, int, int]]:

        triplets = []

        for ds_idx, dataset in enumerate(datasets):
            if ds_idx not in segment_embeddings:
                continue
            
            embeddings = segment_embeddings[ds_idx]
            motion_ids = np.array(dataset.motion_ids[:len(embeddings)])
            
            if len(embeddings) < 3:
                continue
            
            sim_matrix = cosine_similarity(embeddings)

            for i in range(len(embeddings)):
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

                diff_mouse_mask = ~same_mouse_mask
                diff_mouse_sims = sim_matrix[i].copy()
                diff_mouse_sims[~diff_mouse_mask] = -2.0

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
                
                triplets.append((ds_idx, i, pos_idx, neg_idx))
            
            if len(triplets) >= max_triplets:
                break
        
        logger.info(f"[CONTRAIN] Found {len(triplets)} hard triplets (mode={mining_mode}, pos<{pos_threshold:.2f}, neg>{neg_threshold:.2f})")
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
    def _similarity_eval(embeddings) -> Tuple[float, float]:
        pos_sims, neg_sims = [], []
        
        for emb in embeddings.values():
            sim_sub = cosine_similarity(emb)
            n_sub = len(emb)
            
            motion_ids = np.array([i % 2 for i in range(n_sub)])
            
            i_idx, j_idx = np.triu_indices(n_sub, k=1)
            valid_sims = sim_sub[i_idx, j_idx]
            same_mouse = motion_ids[i_idx] == motion_ids[j_idx]
            
            pos_sims.extend(valid_sims[same_mouse].tolist())
            neg_sims.extend(valid_sims[~same_mouse].tolist())
        
        if not pos_sims or not neg_sims:
            logger.warning("[CONTRAIN] Similarity score is empty, using defaults")
            return 0.5, 0.3

        logger.info(f"[CONTRAIN] Same-mouse: {np.mean(pos_sims):.3f} ± {np.std(pos_sims):.3f}")
        logger.info(f"[CONTRAIN] Diff-mouse: {np.mean(neg_sims):.3f} ± {np.std(neg_sims):.3f}")
        
        return np.percentile(pos_sims, 30), np.percentile(neg_sims, 70)

    @staticmethod
    def _silhouette_eval(embeddings):
        silhouette_scores = []

        for emb in embeddings.values():
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(emb)
                score = silhouette_score(emb, labels)
                silhouette_scores.append(score)
            except ValueError:
                continue

        if silhouette_scores:
            mean_score = np.mean(silhouette_scores)
            logger.info(f"[CONTRAIN] Silhouette score = {mean_score:.3f}")
            return mean_score
        else:
            return 0.0