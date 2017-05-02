# state-program-report-tabulate
Python script that tabulates the wide CSV output of state-program-report.

compose_dtypes.py creates a JSON-formatted file containing dtypes for the fields in each fiscal year's data.
Run this script first!

tabulate.py runs from CMD with a single parameter, the fiscal year, as in:

python tabulate.py 2015

params.py contains various explicit dicts and lists necessary to process SPR data. It is not executable and does
not need to be edited.
