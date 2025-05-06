import os
import re
import glob
from tqdm import tqdm
from yt_dlp import YoutubeDL

#region Utils
def sanitize_filename(directory):
		pattern = re.compile(r'[^0-9A-Za-z\uAC00-\uD7A3]')
		for filename in os.listdir(directory):
				old_filename = os.path.join(directory, filename)
				if not os.path.isfile(old_filename):
						continue
				fname, ext = os.path.splitext(filename)
				new_fname = pattern.sub('_', fname).strip('_') + ext
				new_path = os.path.join(directory, new_fname)
				if old_filename != new_path:
						os.rename(old_filename, new_path)
#endregion

class Downloader:
		#region Progress Bar
		class _PBar:
				def __init__(self):
					self.pbar = None

				def __call__(self, task):
						status = task.get('status')
						if status == 'downloading':
								total = task.get('total_bytes') or task.get('total_bytes_estimate')
								downloaded = task.get('downloaded_bytes', 0)
								if total and self.pbar is None:
										filename = os.path.basename(task.get('filename', ''))
										self.pbar = tqdm(total=total, unit='B', unit_scale=True, desc=filename, leave=False)
								if self.pbar:
										self.pbar.total = total
										self.pbar.n = downloaded
										self.pbar.refresh()
								elif status == 'finished':
										if self.pbar:
												self.pbar.close()
												self.pbar = None
												print(f" ✓ Finished: {task.get('filename')}")
		#endregion

		def __init__(self, directory: str):
				self.urls = None
				self.dir = directory
				self.pb_hook = self._PBar()
				self.options = {
						'format': 'bv*+ba/b',
						'retries': 4,
						'quiet': True,
						'writeinfojson': False,
						'progress_hooks': [self.pb_hook]
				}

		def download(self):
				if not os.path.isfile('./urls.txt'):
						print(f"File urls.txt does not exist.")
						return
				if not os.path.exists(self.dir):
						os.makedirs(self.dir)
				with open('urls.txt', 'r') as file:
						lines = [line.strip() for line in file if line.strip()]
				for line in lines:
						try:
								id, url = line.split(',', 1)
						except ValueError:
								print(f"Invalid line in urls.txt: {line}")
								continue
						if glob.glob(os.path.join(self.dir, f"{id}.*")):
								print(f"Video {id} already downloaded. Skipping...")
								continue
						options = self.options.copy()
						options['outtmpl'] = os.path.join(self.dir, f"{id}.%(ext)s")
						with YoutubeDL(options) as ydl:
								try:
										ydl.download([url])
								except Exception as e:
										print(f"Error downloading {url}: {e}")
