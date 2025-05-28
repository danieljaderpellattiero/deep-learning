import os
import time
import math
import logging
import pathlib
from moviepy import VideoFileClip, ColorClip, CompositeVideoClip

logging.getLogger("moviepy").setLevel(logging.ERROR)

MIN_WIDTH, MIN_HEIGHT = 360, 360
MAX_WIDTH, MAX_HEIGHT = 3840, 2160
VALID_RATIOS = [(1, 1), (4, 3), (4, 5), (5, 4), (16, 9), (9, 16)]

categories: list[str] = []

def satisfies_requirements(w: int, h: int) -> bool:
	if not (MIN_WIDTH <= w <= MAX_WIDTH and MIN_HEIGHT <= h <= MAX_HEIGHT):
		return False
	tgt_w, tgt_h = compute_target_dimensions(w, h)
	return (w == tgt_w) and (h == tgt_h)

def compute_target_dimensions(w: int, h: int):
	cw = min(max(w, MIN_WIDTH), MAX_WIDTH)
	ch = min(max(h, MIN_HEIGHT), MAX_HEIGHT)
	ratio_num, ratio_den = min(VALID_RATIOS, key=lambda r: abs((cw/ch) - (r[0]/r[1])))
	k_w = math.ceil(cw / ratio_num)
	k_h = math.ceil(ch / ratio_den)
	k = max(k_w, k_h)
	target_w = ratio_num * k
	target_h = ratio_den * k
	if target_w > MAX_WIDTH or target_h > MAX_HEIGHT:
		k = min(math.floor(MAX_WIDTH / ratio_num), math.floor(MAX_HEIGHT / ratio_den))
		target_w = ratio_num * k
		target_h = ratio_den * k
	return target_w, target_h

def process_video(video_path: pathlib.Path) -> bool:
	clip = VideoFileClip(video_path)
	w, h = clip.size
	if satisfies_requirements(w, h):
		print(f"✔ {os.path.basename(video_path)} skipped: {w}x{h}")
		return False
	print(f"✘ {os.path.basename(video_path)} needs processing: {w}x{h}")
	trg_w, trg_h = compute_target_dimensions(w, h)
	if (w / h) > (trg_w / trg_h):
		clip_resized = clip.resized(width=trg_w)
	else:
		clip_resized = clip.resized(height=trg_h)
	bg = ColorClip(size=(trg_w, trg_h), color=(0, 0, 0), duration=clip_resized.duration)
	clip_padded = CompositeVideoClip([bg, clip_resized.with_position('center', 'center')])
	clip_padded.write_videofile(video_path, codec="libx264", audio_codec="aac",
	                            temp_audiofile="temp-audio.m4a", remove_temp=True, logger=None)
	return True

def init_categories() -> None:
	global categories
	categories = set([file.stem.replace('_', ' ') for file in pathlib.Path('../data/train/urls').glob('*')
	                  if file.is_file()])

if __name__ == "__main__":
	init_categories()
	for split in ["train", "validation", "test"]:
		for category in categories:
			counter: int = 0
			print(f'[{time.strftime("%H:%M:%S")} | {split}, {category}] Processing...')
			video_folder = pathlib.Path(f'../videos/{split}/{category}')
			for video_file_path in video_folder.glob('*'):
				if process_video(video_file_path):
					counter += 1
					print(f'[{time.strftime("%H:%M:%S")} | {split}, {category}] Processed {counter}/'
					f'{len(list(video_folder.glob("*")))} videos')
	exit(0)
