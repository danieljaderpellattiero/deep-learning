import csv
import time
import pathlib
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor, as_completed

from moviepy.config import success

# --------------------------------------------------------------------------- #
#		CONFIGURATION                                                              #
# --------------------------------------------------------------------------- #
youtube = build('youtube', 'v3', developerKey='AIzaSyBQ8JQ54N1rY6wBXqd8ftZAXcL55zIGDpY')
# --------------------------------------------------------------------------- #
#  CONSTANTS                                                                  #
# --------------------------------------------------------------------------- #
YT_API_BATCH: int = 50
DATA_PATH: str = './data/'
METADATA_PATH: str = './metadata/'

# !
def process_tsv_file(split: str, tsv_path: pathlib.Path) -> None:
		category = tsv_path.stem
		output_dir = pathlib.Path(METADATA_PATH) / split / category
		output_dir.mkdir(parents=True, exist_ok=True)
		with tsv_path.open(newline='', encoding='utf-8') as file:
				reader = csv.reader(file, delimiter='\t')
				next(reader, None)
				urls	= [row[0].strip() for row in reader if row and row[0].strip()]
		for url in urls:
				# (how can i manage the success variable?) -> control if fns returns smth not None
				# function call for all the metadata extraction
				# at the end export to hdf5 file

def extract_metadata() ->	None:
		"""
		Extracts YouTube video metadata from TSV files by category and split.
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
				print(f'[{time.strftime("%H:%M:%S")} | {split}] Metadata extraction completed for all categories')

if __name__ == '__main__':
		extract_metadata()
		exit(0)