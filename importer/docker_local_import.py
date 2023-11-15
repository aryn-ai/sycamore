from pathlib import Path
import datetime
import math
import numpy
import os
import requests
import stat
import sys
import time

# ruff: noqa: E402
sys.path.append("../sycamore")
sys.path.append("/app")

import sycamore
from sycamore.functions import HuggingFaceTokenizer
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.extract_table import TextractTableExtractor
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner, HtmlPartitioner

# from simple_config import idx_settings, osrch_args, title_template

running_in_container = False
index = "demoindex0"


# TODO: https://github.com/aryn-ai/sycamore/issues/155 - detect that the opensearch cluster
#       has dropped documents and reload them
# TODO: https://github.com/aryn-ai/sycamore/issues/156 - figure out how to deal with documents
#       that are deleted
# TODO: https://github.com/aryn-ai/sycamore/issues/157 - adjust the way we do importing so
#       that the files have a permanent name so the viewPdf UI option works
# TODO: https://github.com/aryn-ai/sycamore/issues/158 - handle importing problems in a more
#       clever way than blind retry.
def main():
    root_path = "/app/.scrapy"
    if len(sys.argv) <= 1:
        if not os.path.isdir(root_path):
            raise RuntimeError("Missing " + root_path + "; run with single path argument if not in container")
    elif len(sys.argv) != 2:
        raise RuntimeError("Usage: docker_local_injest.py [directory_root]")
    else:
        root_path = sys.argv[1]
        if not os.path.isdir(root_path):
            raise RuntimeError("Missing specified path " + root_path + "; correct argument or remove if in container")

    global running_in_container
    if root_path == "/app/.scrapy":
        print("Assuming execution is in container, using adjusted host")
        running_in_container = True

    if root_path == "/app/.scrapy" and "OPENAI_API_KEY" in os.environ:
        print("WARNING: OPENAI_API_KEY in environment is potentially insecure.")
        # TODO: https://github.com/aryn-ai/sycamore/issues/159 - use docker secrets,
        # if necessary via docker swarm.
        # Then enable the sleep since we shouldn't get the env var any more.

        # print "sleep(300)"
        # time.sleep(300)

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Missing OPENAI_API_KEY")

    if "SYCAMORE_TEXTRACT_PREFIX" not in os.environ:
        raise RuntimeError("Missing SYCAMORE_TEXTRACT_PREFIX (e.g. s3://example or s3://example/dir)")

    if "AWS_ACCESS_KEY_ID" not in os.environ:
        raise RuntimeError("Missing AWS_ACCESS_KEY_ID")

    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        raise RuntimeError("missing AWS_SECRET_ACCESS_KEY")

    if "AWS_SESSION_TOKEN" not in os.environ:
        print("WARNING: AWS_SESSION_TOKEN not present; secret key may not work if it is a short term sso token")

    # TODO: eric - check for AWS_CREDENTIAL_EXIRATION

    root = Path(root_path)

    failures = -1  # special marker for first try
    ray_tasks = get_ray_task_count()
    max_files_per_run = 1

    # Make sure that we spend ~75% of the time running at full parallelism 75% because the tasks
    # should finish in 4 units, and the first 3 of those will be at full parallelism.
    files_per_run_limit = math.floor(ray_tasks * 4)
    if ray_tasks == 1:  # If we only have 1 tasks in parallel, no point in doing more than 1 task/run
        files_per_run_limit = 1

    while True:
        files = find_files(root)
        if len(files) > 0:
            try:
                if len(files) > max_files_per_run:
                    print("Have {} remaining files, too many to process in one run.".format(len(files)))
                    print("Limiting number of files in single run to {}".format(max_files_per_run))
                    numpy.random.shuffle(files)
                    files = files[0:max_files_per_run]
                print("Files:", [str(f["path"]) for f in files], flush=True)
                import_files(root, files)
                time.sleep(1)
                if failures == -1:
                    max_files_per_run = min(ray_tasks * 2, files_per_run_limit)
                    failures = 0
                elif failures > 0:
                    failures = failures - 1
                else:
                    max_files_per_run = min(max_files_per_run * 2, files_per_run_limit)
                print("Successfully imported:", [str(f["path"]) for f in files], flush=True)

                print(
                    "Successful run adjusted failure count to {} and max_files_per_run to {}".format(
                        failures, max_files_per_run
                    )
                )
            except Exception as e:
                if failures == -1:
                    failures = 0
                if max_files_per_run > 1:
                    max_files_per_run = max(1, math.floor(max_files_per_run / 2))
                else:
                    failures = failures + 1
                print(
                    "Failed run adjusted failure count to {} and max_files_per_run to {}".format(
                        failures, max_files_per_run
                    )
                )
                print("WARNING: caught and tolerating exception")
                print("WARNING: exception type:", type(e))
                print("WARNING: exception args:", e.args)
                print("WARNING: exception message:", e)
                sleep_time = min(10**failures, 300)
                print("WARNING: sleep(" + str(sleep_time) + ") in case this is persistent")
                time.sleep(sleep_time)
        else:
            print("No changes at", datetime.datetime.now(), "sleeping", flush=True)
            time.sleep(5)


def get_ray_task_count():
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    usable_mem_bytes = 0.5 * mem_bytes  # use at most half of hosts RAM
    gib = 1024 * 1024 * 1024
    bytes_per_task = 2 * gib  # Importing sort benchmark varies between 1-2GB while running
    ray_tasks = math.floor(usable_mem_bytes / bytes_per_task)
    if ray_tasks <= 0:
        ray_tasks = 1
        print("WARNING: Want 2GiB of RAM/ray-task.")
        print("WARNING: Available memory (50% of total) is only {:.2f} GiB".format(usable_mem_bytes / gib))
        print("WARNING: Will run on single core and hope to not use too much swap")

    if ray_tasks <= 1:
        ray_tasks = 2
        print("WARNING: import_pdf_sort_benchmark requires 2 simultaneous ray tasks to not hang.")
        print("WARNING: on low-memory containers, the logic to adjust the number of files to process")
        print("WARNING: in a batch will clamp the number of files to 1 which will allow for importing")

    return ray_tasks


def find_files(root):
    downloads = root.joinpath("downloads")
    pdf_files = find_recursive(downloads, "pdf")
    html_files = find_recursive(downloads, "html")
    ret = []
    for i in pdf_files + html_files:
        t = reimport_timestamp(root, i["suffix"])
        if t > 0:
            i["timestamp"] = t
            ret.append(i)
        else:
            # print('File unchanged', i['suffix'])
            pass

    return ret


def find_recursive(root, file_type):
    prefix = Path(root)
    type_root = os.path.join(prefix, file_type)
    ret = []
    for path in Path(type_root).rglob("*"):
        ret.append({"path": path, "suffix": path.relative_to(prefix), "type": file_type})

    return ret


def reimport_timestamp(root, suffix):
    s = os.lstat(root.joinpath("downloads", suffix))
    if not stat.S_ISREG(s.st_mode):
        return -1

    file_timestamp = s.st_mtime

    try:
        s = os.lstat(root.joinpath("imported", suffix))
    except FileNotFoundError:
        return file_timestamp

    if s.st_mtime < file_timestamp:
        return file_timestamp

    if s.st_mtime > file_timestamp:
        return file_timestamp

    return 0


def import_files(root, files):
    pending_paths = {}
    for i in files:
        # textractor uses PIL.Image.open to identify images, and it fails with an
        # UnidentifiedImageError on a pdf file that doesn't end in .pdf.
        if i["type"] == "pdf" and not str(i["path"]).endswith(".pdf"):
            print("ERROR: Unable to import", str(i["path"]), "-- pdf files must end in .pdf; fix the crawler")
            continue

        if i["type"] not in pending_paths:
            pending_paths[i["type"]] = []

        pending_paths[i["type"]].append(str(i["path"]))

    if running_in_container:
        wait_for_opensearch_ready()

    if "pdf" in pending_paths:
        import_pdf(pending_paths["pdf"])

    if "html" in pending_paths:
        import_html(pending_paths["html"])

    for i in files:
        imported = root.joinpath("imported", i["suffix"])
        os.makedirs(os.path.dirname(imported), exist_ok=True)
        if not os.path.isfile(imported):
            imported.touch()
        os.utime(imported, (time.time(), i["timestamp"]))


def wait_for_opensearch_ready():
    print("Waiting for opensearch to become ready...", end="")
    # magic port number needs to match the status port from the aryn-opensearch.sh docker script
    while True:
        try:
            r = requests.get("http://opensearch:43477/statusz")
            if r.status_code == 200 and r.text == "OK\n":
                print("Ready")
                return True
            else:
                print("?", end="")
        except requests.exceptions.ConnectionError:
            print("!", end="")
            pass

        print(".", end="", flush=True)
        time.sleep(1)


def import_pdf(paths):
    if len(paths) == 0:
        print("WARNING: import_html called with empty paths")
        return

    openai_llm = OpenAI(OpenAIModels.TEXT_DAVINCI.value)
    tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
    merger = GreedyTextElementMerger(tokenizer, 30)

    ctx = sycamore_init()
    (
        # TODO: eric - implement manifest generation from file
        # so like s3://aryn-datasets-us-east-1/sort_benchmark/manifest.json read by
        # JsonManifestMetadataProvider; same below for HTML importing
        ctx.read.binary(paths, binary_format="pdf", filter_paths_by_extension=False)
        # TODO: eric - figure out how to cache the results of the textract runs so that we can
        # 1) speed up testing; and 2) let people try out sycamore without also having to set up S3
        .partition(
            partitioner=UnstructuredPdfPartitioner(),
            table_extractor=TextractTableExtractor(
                region_name="us-east-1", s3_upload_root=os.environ["SYCAMORE_TEXTRACT_PREFIX"]
            ),
        )
        .merge(merger)
        .extract_entity(
            entity_extractor=OpenAIEntityExtractor(
                "title", llm=openai_llm, prompt_template=get_title_context_template()
            )
        )
        .extract_entity(
            entity_extractor=OpenAIEntityExtractor(
                "authors", llm=openai_llm, prompt_template=get_author_context_template()
            )
        )
        .spread_properties(["path", "title"])
        .explode()
        .embed(
            embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        .write.opensearch(os_client_args=get_os_client_args(), index_name=index, index_settings=get_index_settings())
    )


def sycamore_init():
    ray_tasks = get_ray_task_count()
    print("Using", ray_tasks, "CPUs for execution")
    return sycamore.init(ray_args={"num_cpus": ray_tasks})


def import_html(paths):
    if len(paths) == 0:
        print("WARNING: import_html called with empty paths")
        return

    ctx = sycamore_init()
    (
        ctx.read.binary(paths, binary_format="html", filter_paths_by_extension=False)
        .partition(partitioner=HtmlPartitioner())
        .spread_properties(["path", "title"])
        .explode()
        .embed(
            embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        .write.opensearch(os_client_args=get_os_client_args(), index_name=index, index_settings=get_index_settings())
    )

    # TODO: https://github.com/aryn-ai/sycamore/issues/160 - implement HTML import
    pass


def get_index_settings():
    return {
        "body": {
            "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "dimension": 384,
                        "method": {"engine": "nmslib", "space_type": "l2", "name": "hnsw", "parameters": {}},
                        "type": "knn_vector",
                    },
                    "title": {"type": "text"},
                    "searchable_text": {"type": "text"},
                    "title_embedding": {
                        "dimension": 384,
                        "method": {"engine": "nmslib", "space_type": "l2", "name": "hnsw", "parameters": {}},
                        "type": "knn_vector",
                    },
                    "url": {"type": "text"},
                }
            },
        }
    }


def get_os_client_args():
    args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": False,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }
    if running_in_container:
        args["hosts"][0]["host"] = "opensearch"

    return args


def get_title_context_template():
    # ruff: noqa: E501
    return """
        ELEMENT 1: Jupiter's Moons
        ELEMENT 2: Ganymede 2020
        ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
        ELEMENT 4: From Wikipedia, the free encyclopedia
        ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
        =========
        "Ganymede 2020"

        ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
        ELEMENT 2: Tarun Kalluri * UCSD
        ELEMENT 3: Deepak Pathak CMU
        ELEMENT 4: Manmohan Chandraker UCSD
        ELEMENT 5: Du Tran Facebook AI
        ELEMENT 6: https://tarun005.github.io/FLAVR/
        ELEMENT 7: 2 2 0 2
        ELEMENT 8: b e F 4 2
        ELEMENT 9: ]
        ELEMENT 10: V C . s c [
        ========
        "FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation"

        """


def get_author_context_template():
    # ruff: noqa: E501
    return """
            ELEMENT 1: Jupiter's Moons
            ELEMENT 2: Ganymede 2020
            ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
            ELEMENT 4: From Wikipedia, the free encyclopedia
            ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
            =========
            Audi Laupe, Serena K. Goldberg

            ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
            ELEMENT 2: Tarun Kalluri * UCSD
            ELEMENT 3: Deepak Pathak CMU
            ELEMENT 4: Manmohan Chandraker UCSD
            ELEMENT 5: Du Tran Facebook AI
            ELEMENT 6: https://tarun005.github.io/FLAVR/
            ELEMENT 7: 2 2 0 2
            ELEMENT 8: b e F 4 2
            ELEMENT 9: ]
            ELEMENT 10: V C . s c [
            ========
            Tarun Kalluri, Deepak Pathak, Manmohan Chandraker, Du Tran

            """


####################################

main()
sys.exit(0)
