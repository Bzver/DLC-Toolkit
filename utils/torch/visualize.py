import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple, Union

from utils.helper import fig_to_pixmap, array_to_iterable_runs
from utils.logger import logger


class Embedding_Visualizer:
    def __init__(
        self,
        segment_embeddings: List[np.ndarray],
        segment_motion_ids: List[List[int]],
        segment_frame_indices: List[List[int]],
        visual_labels: List[np.ndarray],
        total_frames: Optional[int] = None
    ):
        self.total_frames = total_frames
    
        self.segment_embeddings = segment_embeddings
        self.segment_motion_ids = segment_motion_ids
        self.segment_frame_indices = segment_frame_indices
        self.n_segments = len(segment_embeddings)

        self.embeddings = np.vstack(segment_embeddings) if segment_embeddings else np.array([])
        self.motion_ids = np.array([m for seg in segment_motion_ids for m in seg])
        self.frame_indices = np.array([f for seg in segment_frame_indices for f in seg])

        self.visual_labels = np.hstack(visual_labels) if visual_labels else np.array([])
        self.segment_visual_labels = visual_labels
    
        self.n_samples = len(self.embeddings)

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

        ax = axes[0, 0]
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.motion_ids, cmap="Set1", alpha=0.6, s=15)
        ax.set_title("By Motion ID", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Motion ID")

        ax = axes[0, 1]
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.frame_indices, cmap="viridis", alpha=0.6, s=15)
        ax.set_title("By Frame Index", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Frame")

        ax = axes[1, 0]
        if self.visual_labels is not None:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.visual_labels, cmap="Set2", alpha=0.6, s=15)
            ax.set_title("By Visual Cluster (K-Means)", fontsize=10)
            plt.colorbar(scatter, ax=ax, label="Cluster")
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

    def plot_agreement_timeline(
        self, 
        stability_window: int = 5,
        dpi: int = 150
    ) -> Tuple[any, List[int]]:

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
        logger.info(f"[VIS] Baseline mapping: Visual Cluster 0→Motion ID {mapping[0]}, 1→{mapping[1]}")

        frame_data = {}
        for seg_idx in range(self.n_segments):
            v_labels = self.segment_visual_labels[seg_idx]
            m_ids = self.segment_motion_ids[seg_idx]
            f_idxs = self.segment_frame_indices[seg_idx]
            
            for i in range(len(m_ids)):
                fid = f_idxs[i]
                if fid not in frame_data:
                    frame_data[fid] = {'motion_ids': [], 'visual_labels': []}
                frame_data[fid]['motion_ids'].append(m_ids[i])
                frame_data[fid]['visual_labels'].append(v_labels[i])
        
        frame_classifications = {}
        for fid, data in frame_data.items():
            motion_ids = np.array(data['motion_ids'])
            visual_labels = np.array(data['visual_labels'])
    
            agrees = 0
            for i in range(len(motion_ids)):
                expected = mapping[visual_labels[i]]
                if expected == motion_ids[i]:
                    agrees += 1
    
            n_instances = len(motion_ids)
            if agrees == n_instances: 
                frame_classifications[fid] = 2
            elif agrees == 0:
                frame_classifications[fid] = 1
            else:
                frame_classifications[fid] = 0

        if self.total_frames is None:
            self.total_frames = max(max(seg) for seg in self.segment_frame_indices) + 1

        agreement_timeline = np.full(self.total_frames, -1.0)
        segment_type_timeline = np.full(self.total_frames, -1, dtype=int)

        for fid, classification in frame_classifications.items():
            if 0 <= fid < self.total_frames:
                segment_type_timeline[fid] = classification
                if fid in frame_data:
                    data = frame_data[fid]
                    agreement_timeline[fid] = len([i for i in range(len(data['motion_ids'])) 
                                                if mapping[data['visual_labels'][i]] == data['motion_ids'][i]]) / len(data['motion_ids'])

        plot_n_export = segment_type_timeline.copy()

        for start, end, val in array_to_iterable_runs(plot_n_export):
            if start == 0 or val != 2:
                continue
            if end - start + 1 < stability_window:
                if start > 0:
                    plot_n_export[start:end+1] = plot_n_export[start-1]

        for start, end, val in array_to_iterable_runs(plot_n_export):
            if start == 0 or end == len(plot_n_export) - 1 or val != 0:
                continue
            if plot_n_export[start-1] == plot_n_export[end+1]:
                plot_n_export[start:end+1] = plot_n_export[start-1]
            elif plot_n_export[start-1] == 1 or plot_n_export[end+1] == 1:
                plot_n_export[start:end+1] = 1

        fig, ax = plt.subplots(figsize=(15, 5))

        sampled_frames = sorted([f for f in frame_classifications.keys() if f in frame_data])
        sampled_agreements = [agreement_timeline[f] for f in sampled_frames if agreement_timeline[f] >= 0]
        
        if sampled_frames and sampled_agreements:
            ax.plot(sampled_frames, sampled_agreements, marker="o", markersize=3, linewidth=1, 
                    label="Frame Agreement", color="navy", alpha=0.6)
        
        colors = {2: "lightgreen", 1: "lightcoral", 0: "lightyellow", -1: "lightgray"}
        labels = {2: "Agree", 1: "Disagree (Swap)", 0: "Ambiguous", -1: "Not Sampled"}
        
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
        if swap_frames:
            ax.vlines(swap_frames, 0, 1, colors="red", linewidth=0.5, alpha=0.5, label="Confirmed Swap")
        
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
        
        return pixmap, swap_frames