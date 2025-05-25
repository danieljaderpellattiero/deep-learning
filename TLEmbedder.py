import os
import h5py
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from twelvelabs import TwelveLabs
from sklearn.metrics.pairwise import cosine_similarity
from twelvelabs.models.embed import EmbeddingsTask, SegmentEmbedding

matplotlib.use('Agg')

#region Utils
def _print_segments(segments: List[SegmentEmbedding]):
		for segment in segments:
				print(
						f"◇ scope={segment.embedding_scope:<10}"
						f" option={segment.embedding_option:<10}"
						f" [{segment.start_offset_sec:.1f}s–{segment.end_offset_sec:.1f}s]"
				)

def _save_segments_hdf5(filename: str, segments: List[SegmentEmbedding], output_path: str):
		filepath = os.path.join(output_path, f"{filename}.hdf5")
		periods: set[float] = {0.0}
		embeddings: dict[str, list[list[float]]] = {"audio": [], "visual-text": []}
		for segment in segments:
				periods.add(segment.end_offset_sec)
				embeddings[segment.embedding_option].append(segment.embeddings_float)
		with h5py.File(filepath, "w") as f:
				periods_group = f.create_group("periods")
				audio_group = f.create_group("audio")
				visual_text_group = f.create_group("visual-text")
				periods_group.create_dataset("periods", data=list(periods))
				audio_group.create_dataset("embeddings", data=embeddings["audio"])
				visual_text_group.create_dataset("embeddings", data=embeddings["visual-text"])
		dimension = len(embeddings["audio"][0]) if embeddings["audio"] else len(embeddings["visual-text"][0])
		print(f"Saved {dimension} segments to {filepath}")

def _plot_similarity(video_ids, audio_similarity, visual_text_similarity):
		sns.set_theme("notebook", style="whitegrid")
		fig, axes = plt.subplots(1, 2, figsize=(14, 8))
		sns.heatmap(audio_similarity,
		            xticklabels=video_ids,
		            yticklabels=video_ids,
		            ax=axes[0],
		            annot=True, fmt=".2f",
		            cbar=True, cbar_kws={'label': 'cosine distance'})
		axes[0].set_title("Audio Similarity (Cosine Distance)")
		sns.heatmap(visual_text_similarity,
		            xticklabels=video_ids,
		            yticklabels=video_ids,
		            ax=axes[1],
		            annot=True, fmt=".2f",
		            cbar=True, cbar_kws={'label': 'cosine distance'})
		axes[1].set_title("Visual-Text Similarity (Cosine Distance)")
		plt.savefig("embedding_similarity.png", dpi=500)
		plt.close(fig)
		print("Saved similarity plot as embedding_similarity.png")
#endregion

class Embedder:
		def __init__(self, api_key: str, input_dir: str, output_dir: str):
				self.embeddings = {}
				self.input_dir = input_dir
				self.output_dir = output_dir
				self.client = TwelveLabs(api_key=api_key)

		def embed_video(self, video_path: str) -> EmbeddingsTask:
				task = self.client.embed.task.create(model_name="Marengo-retrieval-2.7", video_file=video_path)
				print(f"Created task id={task.id!r}; status={task.status!r}")
				task.wait_for_done(sleep_interval=2)
				task = task.retrieve(embedding_option=["visual-text", "audio"])
				return task

		def generate_embeddings(self):
				os.makedirs(self.output_dir, exist_ok=True)
				for filename in sorted(os.listdir(self.input_dir)):
						stem, ext = os.path.splitext(filename)
						hdf5_path = os.path.join(self.output_dir, f"{stem}.hdf5")
						if os.path.exists(hdf5_path):
								print(f"Embedding {stem} already created. Skipping...")
								continue
						task = self.embed_video(os.path.join(self.input_dir, filename))
						if task.video_embedding and task.video_embedding.segments:
								#_print_segments(task.video_embedding.segments)
								_save_segments_hdf5(stem, task.video_embedding.segments, output_path=self.output_dir)
						else:
								print("No segments returned.")

		def load_embeddings(self):
				for filename	in sorted(os.listdir(self.output_dir)):
						with h5py.File(os.path.join(self.output_dir, filename), "r") as file:
								self.embeddings[filename] = {
										"audio": file["audio"]["embeddings"][:],
										"visual-text": file["visual-text"]["embeddings"][:],
										"periods": file["periods"]["periods"][:]
								}

		def calculate_embeddings_similarity(self):
			video_ids = list(self.embeddings.keys())
			audio_features = np.array([])
			visual_text_features = np.array([])
			for vid in video_ids:
					audio_features = np.vstack([audio_features, self.embeddings[vid]["audio"].mean(axis=0)])\
							if audio_features.size else self.embeddings[vid]["audio"].mean(axis=0)
					visual_text_features = np.vstack([visual_text_features, self.embeddings[vid]["visual-text"].mean(axis=0)])\
							if visual_text_features.size else self.embeddings[vid]["visual-text"].mean(axis=0)
			audio_features = audio_features / np.linalg.norm(audio_features, axis=1, keepdims=True)
			visual_text_features = visual_text_features / np.linalg.norm(visual_text_features, axis=1, keepdims=True)
			audio_similarity = cosine_similarity(audio_features)
			visual_text_similarity = cosine_similarity(visual_text_features)
			_plot_similarity(
					[video_ids.rsplit('.')[0] for video_ids in video_ids],
					audio_similarity,
					visual_text_similarity)
