import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.helper import fig_to_pixmap, array_to_iterable_runs


class Embedding_Visualizer:
    def __init__(self, embeddings, motion_ids, frame_indices, visual_labels=None):
        self.embeddings = embeddings
        self.motion_ids = np.array(motion_ids)
        self.frame_indices = np.array(frame_indices)
        self.visual_labels = np.array(visual_labels) if visual_labels is not None else None
        self.n_samples = len(embeddings)

    def plot_tsne_combined(self, perplexity=30, dpi=150):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        embeddings_2d = tsne.fit_transform(self.embeddings)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Embedding Analysis: t-SNE Projections", fontsize=16, fontweight="bold")

        ax = axes[0, 0]
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.motion_ids, cmap="Set1", alpha=0.6, s=15)
        ax.set_title("By Motion ID", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.frame_indices, cmap="viridis", alpha=0.6, s=15)
        ax.set_title("By Frame Index", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        if self.visual_labels is not None:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.visual_labels, cmap="Set2", alpha=0.6, s=15)
            ax.set_title("By Visual Cluster (K-Means)", fontsize=10)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="gray", alpha=0.6, s=15)
            ax.set_title("Visual Cluster (N/A)", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        if self.visual_labels is not None:
            baseline_frames = np.unique(self.frame_indices)[:20]
            cluster_votes = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
            for f in baseline_frames:
                mask = self.frame_indices == f
                for v_cluster in [0, 1]:
                    for m_id in [0, 1]:
                        count = np.sum((self.visual_labels[mask] == v_cluster) & 
                                     (self.motion_ids[mask] == m_id))
                        cluster_votes[v_cluster][m_id] += count
            mapping = {vc: max(cluster_votes[vc], key=cluster_votes[vc].get) for vc in [0, 1]}

            colors = []
            for i in range(self.n_samples):
                expected = mapping[self.visual_labels[i]]
                if expected == self.motion_ids[i]:
                    colors.append(0)  # green
                else:
                    colors.append(1)  # red
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=colors, cmap="RdYlGn", alpha=0.6, s=15)
            ax.set_title("By Agreement Status", fontsize=10)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c="lightgray", alpha=0.6, s=15)
            ax.set_title("Agreement (N/A)", fontsize=10)

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()

        pixmap = fig_to_pixmap(fig, dpi=dpi)
        plt.close(fig)
        
        return pixmap

    def plot_agreement_timeline(self, disagreement_threshold=0.3, stability_window=5, dpi=150):
        if self.visual_labels is None:
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.text(0.5, 0.5, "Visual labels required\nRun K-Means clustering first", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            pixmap = fig_to_pixmap(fig, dpi=dpi)
            plt.close(fig)
            unique_frames = np.unique(self.frame_indices)
            return pixmap, np.zeros(len(unique_frames), dtype=int)
    
        baseline_frames = np.unique(self.frame_indices)[:20]
        cluster_votes = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}

        for f in baseline_frames:
            mask = self.frame_indices == f
            for v_cluster in [0, 1]:
                for m_id in [0, 1]:
                    count = np.sum((self.visual_labels[mask] == v_cluster) & 
                                 (self.motion_ids[mask] == m_id))
                    cluster_votes[v_cluster][m_id] += count
        
        mapping = {}
        for v_cluster in [0, 1]:
            mapping[v_cluster] = max(cluster_votes[v_cluster], key=cluster_votes[v_cluster].get)

        unique_frames = np.unique(self.frame_indices)
        agreement = []
        
        for f in unique_frames:
            mask = self.frame_indices == f
            agrees = 0
            for i in np.where(mask)[0]:
                expected = mapping[self.visual_labels[i]]
                if expected == self.motion_ids[i]:
                    agrees += 1
            agreement.append(agrees / np.sum(mask))
        
        agreement = np.array(agreement)
        segment_type = np.zeros(len(agreement), dtype=int)
        
        for i in range(len(agreement)):
            start_idx = max(0, i - stability_window + 1)
            window_data = agreement[start_idx : i + 1]
            window_mean = np.mean(window_data)
    
            if window_mean >= 1.0 - disagreement_threshold:
                segment_type[i] = 2  # Agree
            elif window_mean <= disagreement_threshold:
                segment_type[i] = 1  # Disagree (Stable Candidate)
            else:
                segment_type[i] = 0  # Ambiguous

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(unique_frames, agreement, marker="o", markersize=3, linewidth=1, 
                label="Raw Agreement", color="navy", alpha=0.6)

        colors = {2:  "lightgreen", 1:  "lightcoral", 0:  "lightyellow"}
        labels = {2:  "Agree", 1: "Disagree (Swap)"}
        
        plot_n_export = np.zeros((np.max(unique_frames)+1), dtype=int)
        plot_n_export[unique_frames.tolist()] = segment_type

        for start, end, val in array_to_iterable_runs(plot_n_export):
            if start == 0:
                continue
            if end - start + 1 < stability_window and val==2:
                plot_n_export[start:end+1] = plot_n_export[start-1]

        for start, end, val in array_to_iterable_runs(plot_n_export):
            if start == 0 or end == len(plot_n_export) - 1 or val != 0:
                continue
            if plot_n_export[start-1] == plot_n_export[end+1]:
                plot_n_export[start:end+1] = plot_n_export[start-1]
            elif plot_n_export[start-1] == 1 or plot_n_export[end+1] == 1:
                plot_n_export[start:end+1] = 1
         
        handled_labels = set([0])
        for start, end, val in array_to_iterable_runs(plot_n_export):
            label = labels[val] if val not in handled_labels else ""
            ax.fill_between(list(range(start,end+1)), 0, 1,  color=colors[val], alpha=0.2, label=label)
            handled_labels.add(val)

        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
        ax.axhline(y=0.0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)

        if np.any(segment_type==1):
            swap_frames = unique_frames[segment_type==1].tolist()
            ax.vlines(swap_frames, 0, 1, colors="red", linewidth=0.5, alpha=0.5, label="Confirmed Swap")

        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Agreement Fraction")
        ax.set_title("Motion Pipeline vs Visual Cluster Agreement Over Time", fontsize=13)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        pixmap = fig_to_pixmap(fig, dpi=dpi)
        plt.close(fig)
        
        return pixmap, np.where(plot_n_export==1)[0].tolist()