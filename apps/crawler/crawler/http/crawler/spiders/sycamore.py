from pathlib import Path

import os
import re
import scrapy
import time
import urllib


class SycamoreSpider(scrapy.Spider):
    name = "sycamore"

    def __init__(self, category=None, *args, **kwargs):
        self.dest_dir = kwargs.get("dest_dir", ".scrapy/downloads")
        self.prefix = None
        if "preset" in kwargs:
            self._setup_by_preset(kwargs["preset"])
        elif "url" in kwargs:
            self.start_urls = [kwargs["url"]]
            if "prefix" in kwargs:
                self.prefix = kwargs["prefix"]
            if "domain" in kwargs:
                self.allowed_domains = [kwargs["domain"]]
            else:
                u = urllib.parse.urlparse(kwargs["url"])
                d = u.hostname.removeprefix("www.")
                print("Using inferred domain", d)
                self.allowed_domains = [d]
        else:
            print(
                "Neither -a preset=<choice> or -a url=... and an optional -a prefix=... or -a domain=... set.\n"
                + "Using default, tiny single document crawl"
            )
            self._setup_by_preset("sort_single")

    def _setup_by_preset(self, preset):
        presets = {
            "sort_single": ["sortbenchmark.org", "http://sortbenchmark.org/2004_Nsort_Minutesort.pdf"],
            "sort": ["sortbenchmark.org", "http://sortbenchmark.org/"],
            "aryn": ["aryn.ai", "http://aryn.ai"],
            "integration_test": ["localhost", "http://localhost:13756"],
        }
        if preset in presets:
            v = presets[preset]
            self.allowed_domains = [v[0]]
            self.start_urls = [v[1]]
        else:
            v = list(presets.keys())
            v.sort()
            raise RuntimeError("Unknown preset " + preset + ". Expected one of: " + " ".join(v))

    def parse(self, response):
        lm = self._last_modified_time_unix(response)
        ct = self._content_type(response)
        name = os.path.join(ct, re.sub("/", "_", response.url))

        file = os.path.join(self.dest_dir, name)
        if ct == "pdf" and not file.endswith(".pdf"):
            file = file + ".pdf"

        if self._possibly_modified(lm, file):
            print("Store", response.url, "as", file)
            os.makedirs(os.path.dirname(file), exist_ok=True)
            Path(file).write_bytes(response.body)
            os.utime(file, (time.time(), lm))

        if ct == "html":
            print("Processing", response.url)
            # Scrapy docs imply this should work, but it doesn't.
            # It misses pdf links on sortbenchmark.org
            # links = LinkExtractor(allow = '^http').extract_links(response)
            links = response.css("a::attr(href)").getall()
            for i in links:
                i = urllib.parse.urljoin(response.url, i)
                if not i.startswith("http"):
                    continue

                if self.prefix is not None and not i.startswith(self.prefix):
                    print("  Skipping", i, "as it does not start with", self.prefix)
                    continue

                print("  Following", i)
                yield scrapy.Request(i, callback=self.parse)

    def _content_type(self, response):
        ctk = "Content-Type"
        if ctk not in response.headers:
            print("No Content-Type ", response.url)
            return "unknown"
        ct = response.headers[ctk]
        if ct.startswith(b"text/html"):
            return "html"
        if ct == b"application/pdf":
            return "pdf"
        print('Unknown content type "', ct, '" in ', response.url)
        return "unknown"

    def _last_modified_time_unix(self, response):
        lm = "Last-Modified"
        if lm not in response.headers:
            print("WARNING: " + response.url + " is missing last-modified header")
            return True
        date = str(response.headers[lm], "UTF-8")
        t = time.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")
        return time.mktime(t)

    def _possibly_modified(self, last, file):
        try:
            m = os.path.getmtime(file)
            return m < last
        except FileNotFoundError:
            return True
