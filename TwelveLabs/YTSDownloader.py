import os
import re
from tqdm import tqdm
from yt_dlp import YoutubeDL

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
		#endregion

		def __init__(self):
				self.urls = None
				self.pb_hook = self._PBar()
				self.options = {
						'format': 'bv*+ba/b',
						'outtmpl': os.path.join('./shorts', '%(title)s.%(ext)s'),
						'retries': 4,
						'quiet': True,
						'writeinfojson': False,
						'progress_hooks': [self.pb_hook]
				}

		def download(self):
				if not os.path.isfile('./urls.txt'):
						print(f"File urls.txt does not exist.")
						return
				else:
						with open('urls.txt', 'r') as f:
								self.urls = [line.strip() for line in f if line.strip()]
				if not os.path.exists('./shorts'):
						os.makedirs('./shorts')
				with YoutubeDL(self.options) as ydl:
						ydl.download(self.urls)
				sanitize_filename('./shorts')
