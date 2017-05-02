#!/usr/bin/env python
# -*- coding: utf-8 -*-

# tabulate.py : Reconstructs a wide CSV of SPR data into three separate CSV files

# Developed by the Office of Digital and Information Strategy at the Institute of Museum & Library Services.
# As a work of the United States Government, this package is in the public domain within the United States.
# Additionally, IMLS waives copyright and related rights in the work worldwide through the CC0 1.0 Universal
# public domain dedication (which can be found at http://creativecommons.org/publicdomain/zero/1.0/).

import glob
import json
import numpy as np
import os
import pandas as pd
import sys
import unicodedata
from HTMLParser import HTMLParser
import params

fiscalyear = str(2015)  # set fiscal year here if running interactively

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

if not os.path.exists(outpath):
	os.makedirs(outpath)

dtypes_file = os.path.join(cwd_path, 'dtypes_' + fy + '.json')

""" DEFINE FUNCTIONS - - - - - - - - - - - - - - - - - - - - - - - - - - """

# Strips html tags from text fields
class MLStripper(HTMLParser):
	def __init__(self):
		HTMLParser.__init__(self)
		self.reset()
		self.fed = []

	def handle_data(self, d):
		self.fed.append(d)

	@property
	def get_data(self):
		return self.fed


def strip_tags(html):
	s = MLStripper()
	s.feed(html)
	return s.get_data


# Merges multiple dicts into a single dict
def merge_dicts(*dict_args):
	result = {}
	for dictionary in dict_args:
		result.update(dictionary)
	return result

""" IMPORT DATA FILE - - - - - - - - - - - - - - - - - - - - - - - - - - """

# Locates the file for the selected fiscal year
fname_part = "Projects-FY"
infolder = os.path.join(generated_path, fy)
infolder_files = os.listdir(infolder)

for f in infolder_files:
	if fname_part in f:
		sourcefile = os.path.join(infolder, f)

# Retrieves dtype values from JSON file
with open(dtypes_file) as jsondata:
	dtypes = json.load(jsondata)

# Reads input CSV to DataFrame
df = pd.read_csv(sourcefile, encoding="utf-8", dtype=dtypes)

""" CLEAN-UP DATA FILE - - - - - - - - - - - - - - - - - - - - - - - - - """

# Removes unnecessary records and sets index
df = df[df.State != 'Ztest']
df = df.copy().set_index('ProjectCode')

# Sort the good from the bad
bad_file = os.path.join(outpath, 'SPR_FY' + fy + '_badStatus.csv')
statuses = list(set(df['Status'].tolist()))
good_status = ['Approved', 'Accepted', 'Certified', 'Completed']
bad_status = [s for s in statuses if s not in good_status]

df_bad = df.loc[df['Status'].isin(bad_status)].copy()
df_bad.to_csv(bad_file, encoding="utf-8")

df = df.loc[df['Status'].isin(good_status)].copy()

# Find and remove any columns that have no values
cols = [c for c in df.columns]
n = []

for c in cols:
	if df[c].isnull().values.all():
		n.append(c)
for x in n:
	del df[x]

del cols

# Replaces 'NaN' with '' in object/str columns
str_cols = [x for x in df.columns if df[x].dtype == 'object']

for s in str_cols:
	df[s].fillna('', inplace=True)

# Strips html tags from all long text fields
long = params.html_fields_p
cols = df.columns.tolist()
rows = df.index.tolist()

strp_flds = []

for col in cols:
	for l in long:
		if col.startswith(l):
			strp_flds.append(col)

for col in strp_flds:
	for row in rows:
		try:
			t = df.loc[row][col]
			u = strip_tags(t)
			v = ''.join(str(element) for element in u)
			df.set_value(row, col, v)
		except:
			pass
df = df.copy()

# Strips non-ASCII characters
stripped = lambda x: "".join(i for i in x if 31 < ord(i) < 127)

for s in str_cols:
	df[s] = df[s].apply(stripped)

# Strips white space
str_strip = lambda x: str(x).strip()

for s in str_cols:
	df[s] = df[s].apply(str_strip)

# Trims field length to 32,750
str_trim = lambda x: x[0:32750]

for s in str_cols:
	df[s] = df[s].apply(str_trim)

# Replaces '.' with '~' in certain field names
editColumns = []

searchList = ['LinkURL', 'ProjectTag', 'IntentName', 'IntentSubject']

for col in df.columns:
	if any(s in col for s in searchList):
		editColumns.append(col.replace('.', '~'))
	else:
		editColumns.append(col)

colPairs = zip(df.columns, editColumns)
colDict = dict([i for i in colPairs if not i[0] == i[1]])
df.rename(columns=colDict, inplace=True)


""" PROJECTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - """
intent_name_nums = [x[-1:] for x in df.columns if "IntentName" in x]
intent_subject_nums = set([x[-1:] for x in df.columns if "IntentSubject" in x])

# Standardizes format of IntentName and IntentSubject fields
unicode_stripper = lambda x: unicodedata.normalize('NFKD', x)\
	.encode('ascii', 'ignore') if isinstance(x, unicode) else x

for n in intent_name_nums:
	if 'IntentName~' + n in df.columns:
		df['IntentName~' + n] = df['IntentName~'+ n]\
								.map(unicode_stripper).str.upper()\
								.str.replace('.', '')\
								.str.rstrip(' ')

for n in intent_name_nums:
	for s in intent_subject_nums:
		if 'IntentSubject~' + n + '~' + s in df.columns:
			df['IntentSubject~' + n + '~' + s] = df['IntentSubject~' + n + '~' + s]\
												 .map(unicode_stripper)\
												 .str.upper()\
												 .str.replace('.', '')\
												 .str.rstrip(' ')


# Joins an external mapping dict of IntentName --> FocalArea
focal_areas = pd.DataFrame.from_dict(params.focalAreas, orient='index')
focal_areas.index.name = 'IntentName'
focal_areas.columns = ['FocalArea']

for n in intent_name_nums:
	df = df.rename(columns={'IntentName~' + n: 'IntentName'}).copy()
	df = df.join(focal_areas, on='IntentName', lsuffix='L_').copy()
	df['FocalArea~' + n] = df['FocalArea'].copy()
	del df['FocalArea']
	df = df.rename(columns={'IntentName': 'IntentName~' + n}).copy()


# Creates a field for use in geocoding
df['Esri_Address'] = (
	df['GranteeAddress1'] + ', ' +
	df['GranteeCity'] + ' ' +
	df['GranteeState'] + ' ' +
	df['GranteeZip'])

# Creates dfP - dataframe with all Project fields
projects_file = os.path.join(outpath, 'SPR_FY' + fy + '_Projects.csv')
cols = [c for c in list(df) if len(c.split('.')) == 1]
dfP = df[cols].copy()
dfP.reset_index(drop=False, inplace=True)

# Re-orders fields to prescribed order
p_order = params.p_order
newcols = [x for x in p_order if x in dfP.columns]
dfP2 = dfP[newcols]

# Exports dfP to 'SPR_FYxxxx_Projects.csv'
dfP2.to_csv(projects_file, encoding='utf-8', index=False)
del newcols


""" ACTIVITIES - - - - - - - - - - - - - - - - - - - - - - - - - - - - - """

# Creates the Activities dataframe
activities_file = os.path.join(outpath, 'SPR_FY' + fy + '_Activities.csv')

# Creates a list of column names
cols = [c for c in list(df) if len(c.split('.')) == 2]

# Creates a list of Activity Numbers
c0 = [c.split('.')[0] for c in cols]
c1 = ['A' + str(c.split('.')[1]) for c in cols]

# Zips column names and Activity Numbers into tuples
arrays = [c0, c1]
tuples = list(zip(*arrays))
# anums.sort(key=lambda x: int(x[1:]))

dfa = pd.DataFrame(df, columns=cols).copy()
dfa.columns = pd.MultiIndex.from_tuples(tuples, names=['field', 'activity'])
dfa = dfa.stack(level=1).copy()
dfa.reset_index(drop=False, inplace=True)
dfa['ActivityCode'] = dfa['ProjectCode'] + '-' + dfa['activity'].str[1:]
dfa = dfa.copy().set_index('ActivityCode')
del dfa['activity']

# Strip HTML from 'ActivityAbstract'
rows = dfa.index.tolist()
fields = params.html_fields_a

for field in fields:
	for row in rows:
		try:
			t = dfa.loc[row]['ActivityAbstract']
			u = ''.join([i if ord(i) < 128 else ' ' for i in t])
			u = strip_tags(t)
			v = ''.join(str(element) for element in u)
			dfa.set_value(row, 'ActivityAbstract', v)
		except:
			pass

dfa = dfa.copy()
dfa = dfa.loc[pd.notnull(dfa['ActivityNumber'])]

# Appends the Outcomes fields to the Activities dataframe
cols = [c for c in list(df) if len(c.split('.')) == 3 and "Quantity" in c]
c0 = [c.split('.')[0] for c in cols]
c1 = ['A' + str(c.split('.')[1]) for c in cols]
c2 = ['Q' + str(c.split('.')[2]) for c in cols]
arrays = [c0, c1, c2]
tuples = list(zip(*arrays))
anums = list(set(c1))
anums.sort(key=lambda x: int(x[1:]))

df1 = pd.DataFrame(df, columns=cols).copy()
df1.columns = pd.MultiIndex.from_tuples(tuples, names=['field',
                                                       'activity',
                                                       'outcome'])
dataFrames = []

for a in anums:
	dfx = df1.iloc[:, df1.columns.get_level_values(1) == a]
	dfy = dfx.stack(level=1).stack(level=1).copy()
	dfy['QuantityName'] = dfy['QuantityName'].map(lambda x: x.strip())
	dataFrames.append(dfy)

df2 = pd.concat(dataFrames, axis=0)
df2.reset_index(drop=False, inplace=True)

df2['ActivityCode'] = df2['ProjectCode'] + '-' + \
                      df2['activity'].str[1:]

df2['OutcomeCode'] = df2['ProjectCode'] + '-' + \
                     df2['activity'].str[1:] + '-' + \
                     df2['outcome'].str[1:]

df3 = df2.copy().set_index('OutcomeCode')
df3 = df3[pd.notnull(df3['QuantityValue'])]
df4 = pd.pivot_table(df3,
                     values='QuantityValue',
                     index='ActivityCode',
                     columns='QuantityName')

df4.reset_index(inplace=True)
dfa.reset_index(inplace=True)
dfa2 = pd.merge(dfa, df4, on='ActivityCode', how='outer')
del df1, df2, df3, df4

# Appends the Partner Organization fields to the Activities dataframe
cols = [c for c in list(df) if len(c.split('.')) == 3 and "PartnerOrg" in c]
c0 = [c.split('.')[0] for c in cols]
c1 = ['A' + str(c.split('.')[1]) for c in cols]
c2 = ['R' + str(c.split('.')[2]) for c in cols]
arrays = [c0, c1, c2]
tuples = list(zip(*arrays))
anums = list(set(c1))
anums.sort(key=lambda x: int(x[1:]))
df1 = pd.DataFrame(df, columns=cols).copy()
df1.columns = pd.MultiIndex.from_tuples(tuples, names=['field',
                                                       'activity',
                                                       'partner'])

dataFrames = []

for a in anums:
	dfx = df1.iloc[:, df1.columns.get_level_values(1) == a]
	dfy = dfx.stack(level=1).stack(level=1).copy()
	dataFrames.append(dfy)

df2 = pd.concat(dataFrames, axis=0)
df2.reset_index(drop=False, inplace=True)

df2['ActivityCode'] = df2['ProjectCode'] + '-' + \
                      df2['activity'].str[1:]

df2['PartnerCode'] = df2['ProjectCode'] + '-' + \
                     df2['activity'].str[1:] + '-' + \
                     df2['partner'].str[1:]

df3 = df2.copy().set_index('PartnerCode')

df3.loc[df3['PartnerOrganizationArea'] != '', 'PartnerOrgAreaY'] = 1
df3.loc[df3['PartnerOrganizationType'] != '', 'PartnerOrgTypeY'] = 1


df3a = pd.pivot_table(df3,
                      values='PartnerOrgAreaY',
                      index='ActivityCode',
                      columns='PartnerOrganizationArea',
                      )

df3b = pd.pivot_table(df3,
                      values='PartnerOrgTypeY',
                      index='ActivityCode',
                      columns='PartnerOrganizationType'
                      )

del df3a['']
del df3b['']

index_a = df3a.index.tolist()
index_b = df3b.index.tolist()

df3a.dropna(how='all', inplace=True)
df3b.dropna(how='all', inplace=True)

df3a.reset_index(inplace=True)
df3b.reset_index(inplace=True)


dfa3 = pd.merge(dfa2,
               df3a,
               on='ActivityCode',
               how='outer'
               )

dfa4 = pd.merge(dfa3,
               df3b,
               on='ActivityCode',
               how='outer'
               )

# Prepends "PartnerArea" to fields derived from the values of PartnerOrganizationArea
area_columns_1 = df3a.columns.tolist()
area_columns_1.remove('ActivityCode')
area_columns_2 = ["PartnerArea~" + a for a in area_columns_1]
area_columns_zip = zip(area_columns_1, area_columns_2)
areaDict = dict([i for i in area_columns_zip])
dfa4.rename(columns=areaDict, inplace=True) #22

# Prepends "PartnerType" to fields derived from the values of PartnerOrganizationType
type_columns_1 = df3b.columns.tolist()
type_columns_1.remove('ActivityCode')
type_columns_2 = ["PartnerType~" + a for a in type_columns_1]
type_columns_zip = zip(type_columns_1, type_columns_2)
typeDict = dict([i for i in type_columns_zip])
dfa4.rename(columns=typeDict, inplace=True) #22

for c in area_columns_2:
	dfa4.loc[pd.isnull(dfa4[c]), c] = 'False'
	dfa4.loc[dfa4[c] == 1, c] = 'True'

for c in type_columns_2:
	dfa4.loc[pd.isnull(dfa4[c]), c] = 'False'
	dfa4.loc[dfa4[c] == 1, c] = 'True'

dfa4.set_index('ActivityCode')

str_cols = [x for x in dfa4.columns if dfa4[x].dtype == 'object']
for s in str_cols:
	dfa4[s].fillna('', inplace=True)

# Re-order fields for export
a_order = params.a_order
newcols = [x for x in a_order if x in dfa4.columns]
dfa5 = dfa4[newcols]

# Exports Activities dataframe to Activities.csv
dfa5.to_csv(activities_file, encoding='utf-8', index=False)


""" LOCALES - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"""

# Creates Locales.csv
locales_file = os.path.join(outpath, 'SPR_FY' + fy + '_Locales.csv')

cols = [c for c in list(df) if len(c.split('.')) == 3
        and "LocaleInstitution" in c]

c0 = [c.split('.')[0] for c in cols]
c1 = ['A' + str(c.split('.')[1]) for c in cols]
c2 = ['L' + str(c.split('.')[2]) for c in cols]

arrays = [c0, c1, c2]
tuples = list(zip(*arrays))
anums = list(set(c1))
anums.sort(key=lambda x: int(x[1:]))

dfl1 = pd.DataFrame(df, columns=cols).copy()
dfl1.columns = pd.MultiIndex.from_tuples(tuples, names=['field',
                                                        'activity',
                                                        'locale'])
for col in dfl1.columns:
	dfl1[col] = dfl1[col].apply(unicode_stripper).copy()

dfl1.to_csv(os.path.join(outpath, 'temp.csv'))

dataFrames = []

for a in anums:
	dfx = dfl1.iloc[:, dfl1.columns.get_level_values(1) == a]
	dfy = dfx.stack(level=1).stack(level=1).copy()
	dataFrames.append(dfy)

dfl2 = pd.concat(dataFrames, axis=0) #dfl4
dfl2.reset_index(inplace=True)

dfl2['LocaleCode'] = dfl2['ProjectCode'] + '-' + \
                     dfl2['activity'].str[1:] + '-' + \
					 dfl2['locale'].str[1:]

dfl2['ActivityCode'] = dfl2['ProjectCode'] + '-' + \
					   dfl2['activity'].str[1:]

df_Locales = dfl2.copy().set_index('LocaleCode')

del df_Locales['activity']
del df_Locales['locale']

subset = ['LocaleInstitutionAddress',
		  'LocaleInstitutionCity',
		  'LocaleInstitutionState',
		  'LocaleInstitutionZip']

df_Locales2 = df_Locales.replace('', np.nan)

df_Locales2.dropna(how='all', subset=subset, inplace=True)

# Creates a field for use in geocoding
df_Locales2['Esri_Address'] = (
	df_Locales2['LocaleInstitutionAddress'] + ', ' +
	df_Locales2['LocaleInstitutionCity'] + ' ' +
	df_Locales2['LocaleInstitutionState'] + ' ' +
	df_Locales2['LocaleInstitutionZip']
)

# Exports Locales dataframe to Locales.csv
df_Locales2.to_csv(locales_file, encoding='utf-8')
