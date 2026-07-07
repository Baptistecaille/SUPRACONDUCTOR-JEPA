from tqdm import tqdm
import argparse

from utils.config import DictAction
from utils.crys_utils import final_generate_one_batch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mp_20')
parser.add_argument('--conf_new', nargs='+', action=DictAction)
args = parser.parse_args()

for turn in tqdm(range(10)):
    final_generate_one_batch(args.dataset, turn)

