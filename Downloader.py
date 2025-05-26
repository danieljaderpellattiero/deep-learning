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

def download_and_clip(url: str, out_path: pathlib.Path) -> bool:
	"""
	Downloads a YouTube video and clips it to a specified duration.

	:param url: YouTube video URL
	:param out_path: Path to save the downloaded and clipped video
	:return: True if successful, False otherwise
	"""
	for attempt in range(0, MAX_DOWNLOAD_RETRIES):
		try:
			with YoutubeDL({
				'format': 'bv*+ba/best',
				'merge_output_format': 'mp4',
				'outtmpl': str(out_path / '%(id)s.%(ext)s'),
				'quiet': True,
				'no_warnings': True,
				'logger': CLogger(muted=True)
			}) as yt:
				info = yt.extract_info(url, download=True)
				video_id = info['id']
				video_ext = info['ext']
				video_path = out_path / f'{video_id}.{video_ext}'
				tmp_video_path = out_path / f'{video_id}_tmp.{video_ext}'
				with VideoFileClip(str(video_path)) as clip:
					sub = clip.subclipped(0, CLIP_DURATION)
					sub.write_videofile(str(tmp_video_path), codec='libx264', audio_codec='aac', logger=None)
					tmp_video_path.replace(video_path)
					return True
		except Exception as e:
			print(f'[{time.strftime("%H:%M:%S")}] Error downloading or clipping video: {e} (#{attempt})')
	return False

def process_tsv_file(split: str, tsv_path: pathlib.Path) -> tuple[bool, str]:
	"""
	Processes a TSV file containing YouTube video URLs.

	:param split: Data split (train, validation, test)
	:param tsv_path: Path to the TSV file containing video URLs
	:return: True if all URLs were processed successfully, False otherwise
	"""
	download_counter = 0
	error_occurred = False
	category = tsv_path.stem
	output_path = pathlib.Path(VIDEO_PATH) / split / category
	output_path.mkdir(parents=True, exist_ok=True)
	existing_videos_ids = set([video.stem for video in pathlib.Path(output_path).glob('*') if video.is_file()])
	with tsv_path.open(newline='', encoding='utf-8') as file:
		print(f'[{time.strftime("%H:%M:%S")} | {split}] Processing TSV file: {tsv_path}')
		reader = csv.DictReader(file, delimiter='\t')
		next(reader, None)
		for row	in reader:
			video_url = row['video_url'].strip()
			video_id = row['video_id'].strip()
			if video_id not in existing_videos_ids:
				if download_and_clip(video_url, output_path):
					download_counter += 1
				else:
					error_occurred = True
	print(f'[{time.strftime("%H:%M:%S")} | {split}] Processed {download_counter} URLs for category "{category}"')
	return error_occurred, category

def launch_threaded_downloader() -> None:
	"""
	Launches the threaded downloader for YouTube videos.
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
				error, category = future.result()
				if error:
					print(f'[{time.strftime("%H:%M:%S")} | {split}] Some URLs failed in category "{category}"')

if __name__ == '__main__':
		launch_threaded_downloader()
		exit(0)
