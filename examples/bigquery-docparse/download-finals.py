from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
import google.cloud.storage as storage
import gzip
import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download files from csv.gz lists")
    parser.add_argument(
        "--csv-files", nargs="*", help="csv.gz files to process (default: all *.csv.gz in current directory)"
    )
    parser.add_argument(
        "--pdf-prefix",
        default=os.path.expanduser(os.getenv("BQ_LOCAL_PDFS", "~/pdfs")),
        help="PDF prefix directory (default: $HOME/pdfs)",
    )

    args = parser.parse_args()

    if not args.csv_files:
        csv_files = list(Path(".").glob("*.csv.gz"))
        if not csv_files:
            print("No csv.gz files found in current directory")
            exit(1)
    else:
        csv_files = [Path(f) for f in args.csv_files]

    print(f"Processing CSV files: {[f.name for f in csv_files]}")
    print(f"PDF prefix: {args.pdf_prefix}")

    # Select uri from example.documents where <condition>; then download the file from bigquery as csv.gz.
    process = Process(args.pdf_prefix)
    for csv_file in csv_files:
        process.run(str(csv_file))


class Process:
    def __init__(self, pdf_prefix):
        self.pdf_prefix = pdf_prefix
        self.bucket_client = None
        self.exists_count = 0
        self.already_count = 0
        self.download_count = 0
        self.futures = []
        self.max_workers = 32
        self.max_queued = 64

    def run(self, csv_path):
        print(f"Processing {csv_path}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with gzip.open(csv_path, "rt") as f:
                for ln in f:
                    ln = self.clean_uri(ln)
                    if ln == "uri":
                        continue
                    assert ln.startswith("gs://"), f"bad {ln}"
                    bucket, object = ln.removeprefix("gs://").split("/", 1)
                    if self.bucket_client is None:
                        self.bucket = bucket
                        self.bucket_client = storage.Client().bucket(bucket)
                    else:
                        assert self.bucket == bucket, f"bad {ln}"
                    dest = f"{self.pdf_prefix}/finals/{object}"
                    # print(f"ERIC {object} {dest}")
                    if self.exists(dest):
                        pass
                    elif self.already(object, dest):
                        pass
                    else:
                        self.download(executor, object, dest)
                    # if self.download_count > 100:
                    #    break
            done, not_done = wait(self.futures, return_when=ALL_COMPLETED)
            assert len(done) == len(self.futures)
            self.process_done(list(done))

        print(f"{self.exists_count} files already existed")
        print(f"{self.already_count} files downloaded already")
        print(f"{self.download_count} downloaded files")

    def clean_uri(self, ln):
        ln = ln.strip()
        if ln.startswith('"'):
            assert ln.endswith('"')
            ln = ln.removeprefix('"').removesuffix('"')
        return ln

    def exists(self, dest):
        if not os.path.exists(dest):
            return False

        self.exists_count += 1
        if (self.exists_count % 5000) == 0:
            print(f"{self.exists_count} already existing files")
        return True

    def already(self, object, dest):
        src = f"{self.pdf_prefix}/{object}"
        if not os.path.exists(src):
            return False
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dest)
        self.already_count += 1
        if (self.already_count % 1000) == 0:
            print(f"{self.already_count} already downloaded files")

        return True

    def download(self, executor, object, dest):
        while len(self.futures) >= self.max_queued:
            done, not_done = wait(self.futures, return_when=FIRST_COMPLETED)
            self.process_done(done)
            self.futures = list(not_done)

        f = executor.submit(self.do_download, object, dest)
        self.futures.append(f)
        return True

    def do_download(self, object, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        bits = self.bucket_client.blob(object).download_as_bytes()
        with open(f"{dest}-tmp", "wb") as g:
            g.write(bits)
        os.rename(f"{dest}-tmp", dest)
        return dest

    def process_done(self, done):
        for f in done:
            dest = f.result()
            assert os.path.exists(dest)
            self.download_count += 1
            if (self.download_count % 100) == 0:
                print(f"{self.download_count} files downloaded")


if __name__ == "__main__":
    main()
