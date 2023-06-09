from itertools import zip_longest
import re
import json
import pandas as pd






data_path = "Generative ChatBot/mentalhealth.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')


# group lines by response pair

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
pairs = list(grouper(lines, 2))

