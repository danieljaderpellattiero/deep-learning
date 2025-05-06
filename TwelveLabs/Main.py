import argparse
from TLEmbedder import Embedder
from YTSDownloader import Downloader

if __name__ == "__main__":
		parser = argparse.ArgumentParser(description='Download shorts from YouTube and embed them using TwelveLabs.')
		parser.add_argument('--api-key', required=True, help='TwelveLabs API key')
		parser.add_argument('--input-dir', default='shorts', help='Directory to save downloaded shorts')
		parser.add_argument('--output-dir', default='embeddings', help='Directory to save embeddings as HDF5 files')
		args = parser.parse_args()

		yt_dlp = Downloader(directory=args.input_dir)
		yt_dlp.download()
		embedder = Embedder(api_key=args.api_key, input_dir=args.input_dir, output_dir=args.output_dir)
		embedder.process()
