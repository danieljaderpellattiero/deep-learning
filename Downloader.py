import csv
import time
import pathlib
from yt_dlp import YoutubeDL
from CustomLogger import CLogger
from moviepy import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
# --------------------------------------------------------------------------- #
#  CONSTANTS                                                                  #
# --------------------------------------------------------------------------- #
CLIP_DURATION: int = 20
DATA_PATH: str = './data/'
VIDEO_PATH: str = './videos/'
MAX_DOWNLOAD_RETRIES: int = 3

def download_and_clip(url: str, out_dir: pathlib.Path) -> bool:
		"""
		Downloads a YouTube video and clips it to a specified duration.

		:param url: YouTube video URL
		:param out_dir: Directory to save the downloaded and clipped video
		:return: True if successful, False otherwise
		"""
		for attempt in range(0, MAX_DOWNLOAD_RETRIES):
				try:
						with YoutubeDL({
								'format': 'bv*+ba/best',
								'merge_output_format': 'mp4',
								'outtmpl': str(out_dir / '%(id)s.%(ext)s'),
								'quiet': True,
								'no_warnings': True,
								'logger': CLogger(muted=True)
						}) as yt:
								info = yt.extract_info(url, download=True)
						video_id = info['id']
						video_ext = info['ext']
						video_path = out_dir / f'{video_id}.{video_ext}'
						tmp_video_path = out_dir / f'{video_id}_tmp.{video_ext}'
						with VideoFileClip(str(video_path)) as clip:
								sub = clip.subclipped(0, CLIP_DURATION)
								sub.write_videofile(str(tmp_video_path), codec='libx264', audio_codec='aac', logger=None)
						tmp_video_path.replace(video_path)
						return True
				except Exception as e:
						print(f'[{time.strftime("%H:%M:%S")}] Error downloading or clipping video: {e} (#{attempt})')
		return False

def process_tsv_file(split: str, tsv_path: pathlib.Path) -> None:
		"""
		Processes a TSV file containing YouTube video URLs.
		Downloads the first video and clips it to a specified duration.

		:param split: Data split (train, validation, test)
		:param tsv_path: Path to the TSV file containing video URLs
		:return: None
		:raises Exception: If the TSV file cannot be processed or if the download fails
		"""
		success: bool = True
		category = tsv_path.stem
		output_path = pathlib.Path(VIDEO_PATH) / split / category
		output_path.mkdir(parents=True, exist_ok=True)
		with tsv_path.open(newline='', encoding='utf-8') as file:
			reader = csv.reader(file, delimiter='\t')
			urls = [row[0].strip() for row in reader if row and row[0].strip()]
		for url in urls:
				success = success and download_and_clip(url, output_path)
				if not success:
						raise Exception(url)
		print(f'[{time.strftime("%H:%M:%S")} | {split}] Processed {len(urls)} URLs for category "{category}"')

def launch_threaded_downloader() -> None:
		"""
		Launches the threaded downloader for YouTube videos.
		Downloads and clips videos from TSV files in the specified splits and categories.
		"""
		for split in ['train', 'validation', 'test']:
				tsv_path = pathlib.Path(DATA_PATH) / split / 'urls'
				if not tsv_path.is_dir():
						print(f'[{time.strftime("%H:%M:%S")}] Missing folder: {tsv_path} – skipped')
						continue
				tsv_files = list(tsv_path.glob('*.tsv'))
				if not tsv_files:
						print(f'[{time.strftime("%H:%M:%S")} | {split}] No TSV files in {tsv_path} – skipped')
						continue
				print(f'[{time.strftime("%H:%M:%S")} | {split}] Found {len(tsv_files)} TSV file(s)')
				with ThreadPoolExecutor(max_workers=len(tsv_files)) as executor:
						promises = [executor.submit(process_tsv_file, split, tsv_file) for tsv_file in tsv_files]
						for future in as_completed(promises):
								try:
										future.result()
								except Exception as e:
										print(f'[{time.strftime("%H:%M:%S")}] Error processing TSV file: {e}')
				print(f'[{time.strftime("%H:%M:%S")} | {split}] Download and clipping completed for all categories')

if __name__ == '__main__':
		launch_threaded_downloader()
		exit(0)
