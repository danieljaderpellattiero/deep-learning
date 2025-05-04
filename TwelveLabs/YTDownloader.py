import os
import re
from ProgressBar import PBar
from yt_dlp import YoutubeDL

#region Preliminary setup
# Progress bar for download status
pb_hook = PBar()
# Folder for downloaded shorts
os.makedirs('shorts', exist_ok=True)
# yt-dlp downloader options
options = {
		# Downloading the best format that contains video; if it doesn't already have audio, download and merge the best audio
		'format': 'bv*+ba/b',
		# Saving location and filename template
		'outtmpl': os.path.join('./shorts', '%(title)s.%(ext)s'),
		# Re-try if interrupted
		'retries': 4,
		# Suppressing console output
		'quiet': True,
		# Writing metadata to a JSON file
		'writeinfojson': True,
		# Log progress
		'progress_hooks': [pb_hook]
}
# Reading URLs from a text file
with open('urls.txt', 'r') as f:
		urls = [line.strip() for line in f if line.strip()]
#endregion

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

#region Main
def main():
		with YoutubeDL(options) as ydl:
				ydl.download(urls)
		sanitize_filename('./shorts')

if __name__ == "__main__":
		main()
#endregion
