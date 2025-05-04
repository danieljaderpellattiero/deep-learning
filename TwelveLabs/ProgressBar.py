import os
from tqdm import tqdm

class PBar:
		def __init__(self):
				self.pbar = None

		def __call__(self, task):
				status = task.get('status')
				if status == 'downloading':
						total = task.get('total_bytes') or task.get('total_bytes_estimate')
						downloaded = task.get('downloaded_bytes', 0)
						if total and self.pbar is None:
								filename = os.path.basename(task.get('filename', ''))
								self.pbar = tqdm(
										total=total,
										unit='B',
										unit_scale=True,
										desc=filename,
										leave=False
								)
						if self.pbar:
								self.pbar.total = total
								self.pbar.n = downloaded
								self.pbar.refresh()
				elif status == 'finished':
						if self.pbar:
								self.pbar.close()
								self.pbar = None
						print(f" ✓ Finished: {task.get('filename')}")
