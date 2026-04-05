import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple

from utils.helper import fig_to_pixmap, array_to_iterable_runs
from utils.logger import logger


class Embedding_Visualizer:
    def __init__(
        self,
        segment_embeddings: List[np.ndarray],
        segment_motion_ids: List[List[int]],
        segment_frame_indices: List[List[int]],
        visual_labels: List[np.ndarray],
        assignment_confidence: List[np.ndarray] = None,
        confidence_threshold: float = 0.2,
        total_frames: Optional[int] = None,
        start_idx: int = 0,
        end_idx: int = -1
    ):
    
        self.segment_embeddings = segment_embeddings
        self.segment_motion_ids = segment_motion_ids
        self.segment_frame_indices = segment_frame_indices
        self.n_segments = len(segment_embeddings)

        self.embeddings = np.vstack(segment_embeddings) if segment_embeddings else np.array([])
        self.motion_ids = np.array([m for seg in segment_motion_ids for m in seg])
        self.frame_indices = np.array([f for seg in segment_frame_indices for f in seg])

        self.visual_labels = np.hstack(visual_labels) if visual_labels else np.array([])
        self.segment_visual_labels = visual_labels
        self.assignment_confidence = assignment_confidence
        self.confidence_flat = np.hstack(assignment_confidence) if assignment_confidence else None
        self.confidence_threshold = confidence_threshold
    
        self.total_frames = total_frames if total_frames else max(max(seg) for seg in self.segment_frame_indices) + 1
        self.n_samples = len(self.embeddings)
        self.start_idx = start_idx
        self.end_idx = self.total_frames if end_idx < 0 else end_idx

        logger.info(f"[VIS] Initialized with {self.n_segments} segments, {self.n_samples} total samples")

    def plot_tsne_combined(self, perplexity=30, dpi=150):
        if self.n_samples < 10:
            logger.warning("[VIS] Too few samples for t-SNE")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, "Insufficient samples for t-SNE", ha="center", va="center")
            ax.axis("off")
            pixmap = fig_to_pixmap(fig, dpi=dpi)
            plt.close(fig)
            return pixmap
        
        tsne = TSNE(n_components=2, perplexity=min(perplexity, self.n_samples - 1), random_state=42, max_iter=1000)
        embeddings_2d = tsne.fit_transform(self.embeddings)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Embedding Analysis: t-SNE Projections", fontsize=16, fontweight="bold")

        sharpness = 10
        alphas = 0.2 + 0.7 * (1 / (1 + np.exp(-sharpness * (self.confidence_flat - self.confidence_threshold))))
        alphas = np.clip(alphas, 0.1, 0.95)
    
        ax = axes[0, 0]
        if self.confidence_flat is not None:
            for mid in [0, 1]:
                mask = self.motion_ids == mid
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=[f"tab:{'blue' if mid==0 else 'orange'}"], alpha=alphas[mask], s=15, label=f"Motion ID {mid}")
        else:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=self.motion_ids, cmap="Set1", alpha=0.6, s=15)
            plt.colorbar(scatter, ax=ax, label="Motion ID")
        
        ax.set_title("By Motion ID", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[0, 1]
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.frame_indices, cmap="viridis", alpha=0.6, s=15)
        ax.set_title("By Frame Index", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Frame")

        ax = axes[1, 0]
        if self.visual_labels is not None:
            if self.confidence_flat is not None:
                for cluster in [0, 1]:
                    mask = self.visual_labels == cluster
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                            c=[f"tab:{'green' if cluster==0 else 'red'}"], 
                            alpha=alphas[mask], s=15, label=f"Cluster {cluster}")
            else:
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=self.visual_labels, cmap="Set2", alpha=0.6, s=15)
                plt.colorbar(scatter, ax=ax, label="Cluster")
            ax.set_title("By Visual Cluster (K-Means)", fontsize=10)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="gray", alpha=0.6, s=15)
            ax.set_title("Visual Cluster (N/A)", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        if self.visual_labels is not None:
            baseline_segments = min(3, self.n_segments)
            cluster_votes = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
            
            for seg_idx in range(baseline_segments):
                v_labels = self.segment_visual_labels[seg_idx]
                m_ids = self.segment_motion_ids[seg_idx]
                
                for v_cluster in [0, 1]:
                    for m_id in [0, 1]:
                        mask = (v_labels == v_cluster) & (np.array(m_ids) == m_id)
                        cluster_votes[v_cluster][m_id] += np.sum(mask)
            
            mapping = {vc: max(cluster_votes[vc], key=cluster_votes[vc].get) for vc in [0, 1]}

            colors = []
            for seg_idx in range(self.n_segments):
                v_labels = self.segment_visual_labels[seg_idx]
                m_ids = self.segment_motion_ids[seg_idx]
                for i in range(len(m_ids)):
                    expected = mapping[v_labels[i]]
                    colors.append(0 if expected == m_ids[i] else 1)
            
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap="RdYlGn", alpha=0.6, s=15)
            ax.set_title("By Agreement Status", fontsize=10)
            plt.colorbar(scatter, ax=ax, ticks=[0, 1], label=['Agree', 'Disagree'])
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="lightgray", alpha=0.6, s=15)
            ax.set_title("Agreement (N/A)", fontsize=10)

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        pixmap = fig_to_pixmap(fig, dpi=dpi)
        plt.close(fig)
        
        return pixmap

    def plot_agreement_timeline(self, dpi: int = 150) -> Tuple[any, List[int]]:
        baseline_segments = min(10, self.n_segments)
        cluster_votes = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}

        for seg_idx in range(baseline_segments):
            v_labels = self.segment_visual_labels[seg_idx]
            m_ids = self.segment_motion_ids[seg_idx]
            for v_cluster in [0, 1]:
                for m_id in [0, 1]:
                    mask = (v_labels == v_cluster) & (np.array(m_ids) == m_id)
                    cluster_votes[v_cluster][m_id] += np.sum(mask)

        mapping = {vc: max(cluster_votes[vc], key=cluster_votes[vc].get) for vc in [0, 1]}
        if mapping[0] == mapping[1]:
            mapping = {0: 0, 1: 1}

        agreement_timeline = np.full(self.total_frames, np.nan)
        for seg_idx in range(self.n_segments):
            v_labels = self.segment_visual_labels[seg_idx]
            m_ids = self.segment_motion_ids[seg_idx]
            f_idxs = self.segment_frame_indices[seg_idx]
            confidences = self.assignment_confidence[seg_idx] if self.assignment_confidence else None

            for i in range(len(m_ids)):
                expected = mapping[v_labels[i]]
                agree = (expected == m_ids[i])
                f = f_idxs[i]

                conf = confidences[i] if confidences is not None else 1.0

                if conf < self.confidence_threshold:
                    continue
                if np.isnan(agreement_timeline[f]):
                    agreement_timeline[f] = agree / 2
                else:
                    agreement_timeline[f] += agree / 2

        segment_agreement_timeline = np.full(self.total_frames, np.nan)
        segment_type_timeline = np.full(self.total_frames, 0, dtype=np.uint8)

        for seg_idx in range(self.n_segments):
            f_idxs = self.segment_frame_indices[seg_idx]
            if np.all(np.isnan(agreement_timeline[f_idxs])):
                continue

            segment_avg = np.nanmean(agreement_timeline[f_idxs])
            if segment_avg > 0.6:
                seg_class = 2
            elif segment_avg < 0.4:
                seg_class = 1
            else:
                seg_class = 0
            segment_agreement_timeline[int(np.mean(f_idxs))] = segment_avg
            segment_type_timeline[f_idxs] = seg_class

        diagnosis_timeline = np.column_stack((agreement_timeline, segment_agreement_timeline, segment_type_timeline))

        valid_mask = ~np.isnan(segment_agreement_timeline)
        x = np.arange(self.total_frames)
        smoothed_timeline = gaussian_filter1d(
            np.interp(x, x[valid_mask], segment_agreement_timeline[valid_mask]), sigma=40.0, mode='nearest')

        plot_n_export = segment_type_timeline.copy()

        for start, end, val in array_to_iterable_runs(plot_n_export):
            if start == 0 or end == len(plot_n_export) - 1 or val != 0:
                continue
            if plot_n_export[start-1] == plot_n_export[end+1]:
                plot_n_export[start:end+1] = plot_n_export[start-1]
            elif plot_n_export[start-1] == 1 or plot_n_export[end+1] == 1:
                plot_n_export[start:end+1] = 1

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(smoothed_timeline[self.start_idx:self.end_idx+1], marker="o", markersize=1, linewidth=1, label="Segment Agreement", color="navy", alpha=0.6)
        
        colors = {2: "lightgreen", 1: "lightcoral", 0: "lightyellow"}
        labels = {2: "Agree", 1: "Disagree (Swap)", 0: "Ambiguous"}
        
        handled_labels = set()
        for start, end, val in array_to_iterable_runs(plot_n_export):
            if val not in colors:
                continue
            label = labels[val] if val not in handled_labels else ""
            ax.fill_between(list(range(start, end+1)), 0, 1, color=colors[val], alpha=0.3, label=label)
            handled_labels.add(val)
        
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
        ax.axhline(y=0.0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        
        swap_frames = np.where(plot_n_export == 1)[0].tolist()

        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Agreement Fraction")
        ax.set_title("Motion Pipeline vs Visual Cluster Agreement Over Time", fontsize=13)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, self.total_frames - 1)
        
        pixmap = fig_to_pixmap(fig, dpi=dpi)
        plt.close(fig)
        
        logger.info(f"[VIS] Identified {len(swap_frames)} frames as swap candidates")
        
        return pixmap, swap_frames, diagnosis_timeline