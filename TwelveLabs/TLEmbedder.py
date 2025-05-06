import os
import h5py
from typing import List
from twelvelabs import TwelveLabs
from twelvelabs.models.embed import EmbeddingsTask, SegmentEmbedding

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
				audio_group = f.create_group("audio")
				visual_text_group = f.create_group("visual-text")
				audio_group.create_dataset("periods", data=list(periods))
				audio_group.create_dataset("embeddings", data=embeddings["audio"])
				visual_text_group.create_dataset("periods", data=list(periods))
				visual_text_group.create_dataset("embeddings", data=embeddings["visual-text"])
		dimension = len(embeddings["audio"][0]) if embeddings["audio"] else len(embeddings["visual-text"][0])
		print(f"Saved {dimension} segments to {filepath}")
#endregion

class Embedder:
		def __init__(self, api_key: str, input_dir: str, output_dir: str):
				self.input_dir = input_dir
				self.output_dir = output_dir
				self.client = TwelveLabs(api_key=api_key)

		def embed_video(self, video_path: str) -> EmbeddingsTask:
				task = self.client.embed.task.create(model_name="Marengo-retrieval-2.7", video_file=video_path)
				print(f"Created task id={task.id!r}; status={task.status!r}")
				task.wait_for_done(sleep_interval=2)
				task = task.retrieve(embedding_option=["visual-text", "audio"])
				return task

		def process(self):
				os.makedirs(self.output_dir, exist_ok=True)
				with open('urls.txt', 'r') as file:
						lines = [line.strip() for line in file if line.strip()]
				for line in lines:
						try:
								id, url = line.split(',', 1)
						except ValueError:
								print(f"Invalid line in urls.txt: {line}")
								continue
						hdf5_path = os.path.join(self.output_dir, f"{id}.hdf5")
						if os.path.exists(hdf5_path):
								print(f"Embedding {id} already created. Skipping...")
								continue
						task = self.embed_video(os.path.join(self.input_dir, f"{id}.*"))
						if task.video_embedding and task.video_embedding.segments:
								#_print_segments(task.video_embedding.segments)
								_save_segments_hdf5(id, task.video_embedding.segments, output_path=self.output_dir)
						else:
								print("No segments returned.")
