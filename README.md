This is a coding exercide covering the following scenario:
You are a freelance analytics consultant who has partnered with the TTPD (i.e., the Tiny Town Police Department) to analyze speeding tickets that have been given to the adult citizens of Tiny Town over the 2020-2023 period.

Inside the folder, ttpd_data, you will find a directory of data for Tiny Town. This dataset will need to be "ingested" for analysis.
The solutions must use the Spark DataFrame API.
You will need to ingest this data into a PySpark environment and answer the following questions for the TTPD.

Hint: Some of the police speed radar detectors have reported issues storing the results in the central database. Make sure the readings are accurate before using them in your solutions.

Hint: According to the official town records, no two citizens actually share the same name in Tiny Town.

The solution uses pyspark and prints out the answers to the following questions:
1. Question 1: Which police officer was handed the most speeding tickets?
2. Question 2: What 3 months (year + month) had the most speeding tickets?  also: What overall month-by-month or year-by-year trends, if any,are seen.
3. Question 3: Who are the top 10 people who have spent the most money paying speeding tickets overall?

Included is also a pytest suite that can be run using:  pytest test_tiny_town.py -v
