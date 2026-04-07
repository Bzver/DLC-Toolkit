import os
import datetime
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

    def train(
        self, 
        dataset: List[Crop_Dataset],
        emp: Emb_Params,
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
        sim_array, margin_array = self._similarity_eval(embeddings, dataset)
        mean_score, p10_score, _ = self._silhouette_eval(embeddings)
        mean_margin = np.nanmean(margin_array)
        p10_margin = np.nanpercentile(margin_array, 10.0)

        if self._check_early_stopping(emp, embeddings, mean_margin, mean_score, p10_margin, p10_score):
            self.model.eval()
            return self.model

        self._train_hard(dataset, optimizer, emp, embeddings, sim_array, skip_easy)
        
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
            datasets_to_skip: List[int] = [],
            batch_size: int = 128,
    ) -> Dict[int, np.ndarray]:

        segment_embeddings = {}
        self.model.eval()
        
        for ds_idx, ds in enumerate(datasets):
            if ds_idx in datasets_to_skip:
                segment_embeddings[ds_idx] = np.full(1, np.nan)
                continue

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

    def _compute_segment_prototypes(
        self,
        datasets: List[Crop_Dataset],
        segment_embeddings: Dict[int, np.ndarray],
        min_samples_per_proto: int = 5,
        min_separation_score: float = 0.2
    ) -> Dict[int, Dict[int, np.ndarray]]:
    
        prototypes = {}
        
        for ds_idx, dataset in enumerate(datasets):
            if ds_idx not in segment_embeddings:
                continue
                
            embeddings = segment_embeddings[ds_idx]
            motion_ids = np.array(dataset.motion_ids[:len(embeddings)])
            
            if len(embeddings) < min_samples_per_proto * 2:
                continue

            centroids = []
            valid_segment = True
            for mid in [0, 1]:
                mask = motion_ids == mid
                if np.sum(mask) >= min_samples_per_proto:
                    centroids.append(np.mean(embeddings[mask], axis=0))
                else:
                    valid_segment = False
                    break
            
            if not valid_segment:
                continue
                
            proto_0, proto_1 = np.array(centroids[0]), np.array(centroids[1])

            sim = cosine_similarity([proto_0], [proto_1])[0, 0]
            separation_score = 1.0 - sim 
    
            if separation_score >= min_separation_score:
                prototypes[ds_idx] = {0: proto_0, 1: proto_1}
                logger.debug(f"[PROTO] Seg {ds_idx}: Separation={separation_score:.3f}")
            else:
                logger.debug(f"[PROTO] Seg {ds_idx}: Skipped (Low Separation={separation_score:.3f})")
                
        logger.info(f"[PROTO] Valid prototypes: {len(prototypes)}/{len(datasets)}")
        return prototypes

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
            embeddings: Dict[int, np.ndarray],
            sim_array: np.ndarray,
            skip_easy: bool = False
    ):

        hard_triplets = self._mine_hard_triplets(
            datasets, embeddings, sim_array[:, 1, 0], sim_array[:, 2, 1], max_triplets=emp.triplets, mining_mode="semihard"
        )

        if len(hard_triplets) == 0:
            logger.warning("[HARDTRAIN] No hard triplets found. Model may already be well-separated.")
            logger.warning("[HARDTRAIN] Skipping hard training phase, proceeding with current model state.")
            return
        
        eval_interval = 5

        logger.info(f"[CONTRAIN] Training on hard triplets...")
        self.model.train()

        curr_miner = "hard" if skip_easy else "semihard"

        best_loss = 1e6
        pleatau_train = 0

        last_eval = emp.warmup - 1
        best_eval = 0.0
        pleatau_eval = 0

        datasets_to_skip = []

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

                embeddings = self._extract_embeddings_list(datasets)
                sim_array, margin_array = self._similarity_eval(embeddings, datasets)
                mean_score, p10_score, _ = self._silhouette_eval(embeddings)
                mean_margin = np.nanmean(margin_array)
                p10_margin = np.nanpercentile(margin_array, 10.0)

                if self._check_early_stopping(emp, embeddings, mean_margin, mean_score, p10_margin, p10_score):
                    logger.info("[HARDTRAIN] Thresholds met on convergence. Stopping early.")
                else:
                    logger.info("[HARDTRAIN] Model converged to best possible state. Stopping early.")
                break 
            
            best_loss = min(curr_loss, best_loss) if curr_loss > 0 else best_loss
            eval_needed = (epoch - last_eval) % eval_interval == 0

            if eval_needed or epoch == emp.epochs - 1:
                last_eval = epoch

                embeddings = self._extract_embeddings_list(datasets, datasets_to_skip)
                sim_array, margin_array = self._similarity_eval(embeddings, datasets)
                mean_score, p10_score, sil_array = self._silhouette_eval(embeddings)

                good_segment_indices = np.where((margin_array > emp.margin) & (sil_array > emp.sil))[0].tolist()
                datasets_to_skip.extend(np.random.choice(good_segment_indices, size=int(0.9*len(good_segment_indices)), replace=False))

                mean_margin = np.nanmean(margin_array)
                p10_margin = np.nanpercentile(margin_array, 10.0)
                eval_score = min(mean_margin, mean_score)

                eval_patience = max(3, emp.pleatau//4)
                if eval_score - best_eval < emp.min_imp:
                    pleatau_eval += 1
                    logger.info(f"[HARDTRAIN] Current eval pleatau: {pleatau_eval}, last improvement: {eval_score - best_eval:.3f}, patience: {eval_patience}")
                else:
                    pleatau_eval = 0
                if len(datasets_to_skip) > len(datasets) * 0.95:
                    logger.info("[HARDTRAIN] Most segments are good. Stopping early.")
                    break
                if self._check_early_stopping(emp, embeddings, mean_margin, mean_score, p10_margin, p10_score):
                    logger.info("[HARDTRAIN] Thresholds met on convergence. Stopping early.")
                    break
                if epoch == emp.epochs - 1:
                    logger.info("[HARDTRAIN] Max epoch reached. Stopping...")
                    break
                if pleatau_eval >= eval_patience:
                    logger.info("[HARDTRAIN] Model converged to best possible state. Stopping early.")
                    break

                logger.info(f"[HARDTRAIN] Updating hard triplets at epoch {epoch+1}, skipping {len(datasets_to_skip)} segments...")
                result = self._mine_hard_triplets(
                    datasets, embeddings, sim_array[:, 1, 0], sim_array[:, 2, 1], max_triplets=emp.triplets, mining_mode=curr_miner, segments_to_skip=datasets_to_skip)
                num_result = len(result)
                if num_result < 10:
                    logger.info(f"[HARDTRAIN] Hard triplets too few ({num_result}, stopping...")
                    break

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
            pos_thresholds: np.ndarray,
            neg_thresholds: np.ndarray,
            max_triplets: int = 5000,
            segments_to_skip: List[int] = [],
            mining_mode: str = "semihard"
    ) -> List[Tuple[int, int, int, int]]:

        triplets = []

        for ds_idx, dataset in enumerate(datasets):
            if ds_idx not in segment_embeddings:
                continue
            if ds_idx in segments_to_skip:
                continue
    
            pos_threshold = pos_thresholds[ds_idx]
            neg_threshold = neg_thresholds[ds_idx]
            
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
                
                triplets.append((ds_idx, i, pos_idx, neg_idx))
            
            if len(triplets) >= max_triplets:
                break
        
        logger.info(f"[CONTRAIN] Found {len(triplets)} hard triplets (mode={mining_mode}, avg_pos_thresh={np.mean(pos_thresholds):.2f}, avg_neg_thresh={np.mean(neg_thresholds):.2f})")
        return triplets

    def _check_early_stopping(self, emp, embeddings, mean_margin, mean_score, p10_margin, p10_score):
        if mean_margin > emp.margin and mean_score >= emp.sil:
            logger.info(
                f"[CONTRAIN] Separation score: {mean_margin} >= {emp.margin} (p10: {p10_margin}); sihouette score: {mean_score} >= {emp.sil}. (p10: {p10_score})")
            if p10_margin >= emp.margin * 0.8 and p10_score >= emp.sil * 0.8:
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
    def _similarity_eval(embeddings:Dict[int, np.ndarray], datasets, suppress_verbose=False) -> Tuple[np.ndarray, np.ndarray]:
        sim_array = np.full((len(embeddings), 3, 2), np.nan)

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
    
            for k, sim in enumerate([pos_sim, neg_sim]):
                sim_array[ds_idx, 0, k] = np.mean(sim)
                sim_array[ds_idx, 1, k] = np.percentile(sim, 20)
                sim_array[ds_idx, 2, k] = np.percentile(sim, 80)

        margin_array = sim_array[:, 1, 0] - sim_array[:, 2, 1]
        
        if not suppress_verbose:
            logger.info(f"[CONTRAIN] Same-mouse: {np.nanmean(sim_array[:, 0, 0]):.3f} ± {np.nanstd(sim_array[:, 0, 0]):.3f} (p10: {np.nanpercentile(sim_array[:, 0, 0], 10)})")
            logger.info(f"[CONTRAIN] Diff-mouse: {np.nanmean(sim_array[:, 0, 1]):.3f} ± {np.nanstd(sim_array[:, 0, 1]):.3f} (p90: {np.nanpercentile(sim_array[:, 0, 1], 90)})")
        
        return sim_array, margin_array

    @staticmethod
    def _silhouette_eval(embeddings:Dict[int, np.ndarray], suppress_verbose=False):
        sil_array = np.full(len(embeddings), np.nan)

        for ds_idx, emb in embeddings.items():
            if emb.size == 1:
                continue
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(emb)
                score = silhouette_score(emb, labels)
                sil_array[ds_idx] = score
            except ValueError:
                continue

        mean_score = np.nanmean(sil_array)
        std_score = np.nanstd(sil_array)
        p10_score = np.nanpercentile(sil_array, 10.0)
        if not suppress_verbose:
            logger.info(f"[CONTRAIN] Silhouette score = {mean_score:.3f} ± {std_score:.3f} (p10: {p10_score:.3f})")
        return mean_score, p10_score, sil_array
    
    @staticmethod
    def _biomodal_eval(embeddings:Dict[int, np.ndarray]):

        all_embeddings = np.vstack([emb for emb in embeddings.values() if emb.size > 1])

        kmeans = MiniBatchKMeans(
            n_clusters=2,
            random_state=42,
            batch_size=1024,
            n_init=5,
            max_iter=300,
            )
        kmeans.fit_predict(all_embeddings)
        centroids = kmeans.cluster_centers_
        dist_to_c0 = np.linalg.norm(all_embeddings - centroids[0], axis=1)
        dist_to_c1 = np.linalg.norm(all_embeddings - centroids[1], axis=1)

        assign_margin = np.abs(dist_to_c0 - dist_to_c1)
        distances = np.concatenate([dist_to_c0, dist_to_c1])

        result = Contrastive_Trainer._compute_bimodality_score(distances)
        confidence_cv = np.mean(assign_margin) / (np.std(assign_margin) + 1e-6)

        logger.info(
            f"Biomodal={result['bimodal']}, "
            f"Bimodal Score={result['score']:.3f}, "
            f"Peak Ratio={result['peak_ratio']:.3f}, "
            f"Center Deviation={result['center_deviation']:.3f}, "
            f"Valley Depth={result['valley_depth']:.3f}, "
            f"CV={confidence_cv:.3f}"
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