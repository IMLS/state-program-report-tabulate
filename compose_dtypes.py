#!/usr/bin/env python
# -*- coding: utf-8 -*-

# compose_dtypes.py : Retrieves column names and dtypes of a file; reassigns dtypes according to prescribed lists.

# Developed by the Office of Digital and Information Strategy at the Institute of Museum & Library Services.
# As a work of the United States Government, this package is in the public domain within the United States.
# Additionally, IMLS waives copyright and related rights in the work worldwide through the CC0 1.0 Universal
# public domain dedication (which can be found at http://creativecommons.org/publicdomain/zero/1.0/).

import sys
import os
import pandas as pd
import glob
import json
import params  # <----------------------------- local_git_path.py containing the local path to the Git location

fiscalyear = str(2014)  # set fiscal year here if running interactively

# Establishes fiscal year to process
if not os.path.exists(sys.argv[1]):
	fy = fiscalyear
else:
	fy = str(sys.argv[1])

# Sets path variables
cwd_path = os.getcwd()
git_path = os.path.dirname(os.getcwd())

generated_path = os.path.join(git_path,
                              'state-program-report',
                              'generated'
                              )

folders = glob.glob(generated_path + "/*")
outpath = os.path.join(cwd_path, 'tabulated')

# Merges multiple dicts into a single dict
def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


# Locates the file for the selected fiscal year
fname_part = "Projects-FY"
infolder = os.path.join(generated_path, fy)
infolder_files = os.listdir(infolder)
for f in infolder_files:
	if fname_part in f:
		sourcefile = os.path.join(infolder, f)

# Reads input CSV to DataFrame
df = pd.read_csv(sourcefile, encoding="utf-8")

# Removes unnecessary records and sets index
df = df[df.State != 'Ztest']
df = df.copy().set_index('ProjectCode')

# Find and remove any columns that have no values
cols = [c for c in df.columns]
n = []
for c in cols:
	if df[c].isnull().values.all():
		n.append(c)
for x in n:
	del df[x]
del cols

# Make a list of column names in the df
cols = df.columns.tolist()

# Categorize fieldnames by desired dtype
d_str = params.dtype_str
dict_str = {}
for c in cols:
	for s in d_str:
		if c.startswith(s):
			dict_str[c] = "str"

d_float = params.dtype_float
dict_float = {}
for c in cols:
	for f in d_float:
		if c.startswith(f):
			dict_float[c] = "float64"

dict_dtypes = merge_dicts(dict_str, dict_float)

# Add 'States'
dict_dtypes['State'] = 'str'

# Check for any fields missed
i = dict_dtypes.items()
i_list = [x[0] for x in i]
missing = []
for c in cols:
	if not c in i_list:
		missing.append(c)

if len(missing) == 0:
	print "No missing fields!"
else:
	print "Missing fields: " + missing

# Export dtypes to JSON file
jsonfile = os.path.join(cwd_path, 'dtypes_' + fy + '.json')
with open(jsonfile, 'w') as outfile:
	json.dump(dict_dtypes, outfile, indent=4, sort_keys=True)


