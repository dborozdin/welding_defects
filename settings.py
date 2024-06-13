from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGES = [IMAGES_DIR / 'SampleV1_1_mp4-1_jpg.rf.3f50c974a91c4e6348dd49491f06def8.jpg',
                  IMAGES_DIR / 'SampleV1_1_mp4-42_jpg.rf.6e68d9186e630ffb996233ad2a593f51.jpg',
                  IMAGES_DIR / 'SampleV1_2_mp4-14_jpg.rf.4dba7e8bd84314a155dd85df33b5f4d9.jpg']

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'welding.pt'

