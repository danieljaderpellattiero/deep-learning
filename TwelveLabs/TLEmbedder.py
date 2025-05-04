import os
from tqdm import tqdm
from typing import List, Optional
from twelvelabs import TwelveLabs
from twelvelabs.models.embed import EmbeddingsTask, SegmentEmbedding

#region Utils
def _print_segments(
	segments: List[SegmentEmbedding],
		max_elements: int = 5
):
		for seg in segments:
				print(
						f"• scope={seg.embedding_scope:<12}"
						f" option={seg.embedding_option:<12}"
						f" [{seg.start_offset_sec:.1f}s–{seg.end_offset_sec:.1f}s]"
				)
				print("  embeddings:", seg.embeddings_float[:max_elements], "…")
#endregion

class Embedder:
		#region Progress Bar
		class _ProgressBar:
				def __init__(self):
						self.pbar = tqdm(total=100, desc="Embedding", unit="%")

				def __call__(self, task: EmbeddingsTask):
						raw = getattr(task, "progress", None) or getattr(task, "done_percent", None)
						if raw is not None:
								pct = int(raw * 100) if raw <= 1 else int(raw)
								delta = pct - self.pbar.n
								if delta > 0:
										self.pbar.update(delta)
						else:
								self.pbar.update(1)
						if task.status.lower() in ("done", "completed"):
								if self.pbar.n < 100:
										self.pbar.update(100 - self.pbar.n)
								self.pbar.close()
		#endregion

		def __init__(
				self,
				api_key: str,
				input_dir: str = "shorts",
		):
				self.client = TwelveLabs(api_key=api_key)
				self.input_dir = input_dir

		def embed_video(self, video_path: str) -> EmbeddingsTask:
				task = self.client.embed.task.create(
						model_name="Marengo-retrieval-2.7",
						video_path=video_path
				)
				print(f"\nCreated task id={task.id!r} status={task.status!r}")
				hook = self._ProgressBar()
				task.wait_for_done(
						sleep_interval=2,
						callback=hook
				)
				task = task.retrieve(embedding_option=["visual-text", "audio"])
				print("Embedding complete.\n")
				return task

		def embed_all(self):
				for filename in sorted(os.listdir(self.input_dir)):
						path = os.path.join(self.input_dir, filename)
						if not os.path.isfile(path):
								continue
						print(f"Embedding {filename} …")
						task = self.embed_video(path)
						if task.video_embedding and task.video_embedding.segments:
								_print_segments(task.video_embedding.segments)
						else:
								print("No segments returned.")