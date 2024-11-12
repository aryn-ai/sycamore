#!/usr/bin/env python

# This script reads a CSV file from an NTSB CAROL query and fetches the PDFs for each report
# listed in the CSV file to the output directory.
#
# To use:
# 1. Run a CAROL Aviation Search: https://www.ntsb.gov/Pages/AviationQueryv2.aspx
#    It's recommended you select "United States" in the "Country" dropdown and select "Completed"
#    in the "Report Status" dropdown.
# 2. Click the "CSV" link under the search results to download a CSV of the results.
# 3. Run this script, supplying the path to the CSV file and the output directory.
#     poetry run python queryui/fetchdata.py --destination output ntsb-results.csv

import argparse
import csv
import logging
import os

import requests

logging.basicConfig(level=logging.INFO)


def main():
    argparser = argparse.ArgumentParser(prog="fetchdata")
    argparser.add_argument("--destination", default="output", help="Output directory")
    argparser.add_argument("--limit", default=None, type=int, help="Limit the number of documents to fetch")
    argparser.add_argument("input", help="Input CSV file", type=argparse.FileType("r"))
    args = argparser.parse_args()

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    reader = csv.DictReader(args.input)
    count = len([name for name in os.listdir(args.destination) if os.path.isfile(os.path.join(args.destination, name))])
    print(f"Starting with {count} existing reports")
    for row in reader:
        try:
            if args.limit and count >= args.limit:
                break
            if not row.get("DocketUrl"):
                print(f"Skipping {row['Mkey']} because it has no DocketUrl")
                continue
            mkey = row["Mkey"]
            if os.path.exists(os.path.join(args.destination, f"{mkey}.pdf")):
                print(f"Skipping {args.destination}/{mkey}.pdf because it already exists")
                continue
            url = f"https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain/GenerateNewestReport/{mkey}/pdf"
            pdf_data = requests.get(url, timeout=60).content
            with open(os.path.join(args.destination, f"{mkey}.pdf"), "wb") as f:
                f.write(pdf_data)
            print(f"Wrote {len(pdf_data)} bytes to {args.destination}/{mkey}.pdf")
            count += 1
        except Exception as e:
            print(f"Error fetching report {mkey}: {e}")

    print(f"Wrote {count} reports to {args.destination}") 


if __name__ == "__main__":
    main()
