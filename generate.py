import argparse
import re
import pickle
from os.path import exists
from train import TextGenerator


class WrongFormatError(Exception):
    def __init__(self, *arg):
        if args:
            self.message = arg[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'WrongFormatError, {0} '.format(self.message)
        else:
            return 'WrongFormatError has been raised'


parser = argparse.ArgumentParser(description='A text generator based on an N gram model')
parser.add_argument('-m', '--model', type=str, default='data/model.pickle', help='Model location')
parser.add_argument('-l', '--length', type=int, default=100, help='Length of the generated sentence')
parser.add_argument('-p', '--prefix', type=str, default="", help='The beginning of the generated text')
parser.add_argument('-a', '--advance', type=int, default=0, help='Use the tag generation model')
args = parser.parse_args()

model_path = args.model
prefix = args.prefix if len(args.prefix) > 0 else None
length = args.length if args.length > 0 else 100
advance = True if args.advance > 1 else False

if not exists(model_path):
    raise FileNotFoundError("The file with the model does not exist at this address")
if re.match(r"[A-Za-zА-Яа-я0-9.\s\\/]+.pickle$", model_path) is None:
    raise WrongFormatError("The file with the model must have the extension .pickle")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

model.set_seed(2025)

print(model.generate(prefix, length, advance))
