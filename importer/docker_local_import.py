from pathlib import Path
import math
import os
import requests
import stat
import sys
import time

# ruff: noqa: E402
sys.path.append("../sycamore")
sys.path.append("/app")

import sycamore
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.embed import SentenceTransformerEmbedder

from simple_config import idx_settings, osrch_args, title_template

running_in_container = False


# TODO: eric - detect that the opensearch cluster has dropped documents and reload them
# TODO: eric - figure out how to deal with documents that are deleted
# TODO: eric - adjust the way we do importing so that the files have a permanent name so the viewPdf UI option works
# TODO: eric - detect that we have insufficient memory and give up if we just keep ooming.
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
        osrch_args["hosts"][0]["host"] = "opensearch"

    if root_path == "/app/.scrapy" and "OPENAI_API_KEY" in os.environ:
        print("WARNING: OPENAI_API_KEY in environment is potentially insecure.")
        # TODO: eric - switch over to docker swarm so we can use docker secrets
        # Then enable the sleep since we shouldn't get the env var any more.
        # print "sleep(300)"
        # time.sleep(300)

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Missing OPENAI_API_KEY")

    root = Path(root_path)

    failures = 0
    loaded_models = False
    while True:
        files = find_files(root)
        print("Files:", files)
        if len(files) > 0:
            try:
                if not loaded_models:
                    print("First time trying to run, only running on a single file so model loading is less flaky")
                    files = files[0:1]
                import_files(root, files)
                time.sleep(1)
                failures = 0
                loaded_models = True
            except Exception as e:
                failures = failures + 1
                print("WARNING: caught and tolerating exception")
                print("WARNING: exception type:", type(e))
                print("WARNING: exception args:", e.args)
                print("WARNING: exception message:", e)
                sleep_time = min(10**failures, 300)
                print("WARNING: sleep(" + str(sleep_time) + ") in case this is persistent")
                time.sleep(sleep_time)
        else:
            print("No changes, sleeping")
            time.sleep(5)


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
    print("Importing PDF files:", paths)
    index = "demoindex0"

    davinci_llm = OpenAI(OpenAIModels.TEXT_DAVINCI.value)

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

    print("Using", ray_tasks, "CPUs for execution")
    ctx = sycamore.init(ray_args={"num_cpus": ray_tasks})
    ds = (
        ctx.read.binary(paths, binary_format="pdf")
        .partition(partitioner=UnstructuredPdfPartitioner())
        .extract_entity(
            entity_extractor=OpenAIEntityExtractor("title", llm=davinci_llm, prompt_template=title_template)
        )
        .spread_properties(["path", "title"])
        .explode()
        .embed(embedder=SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2", batch_size=100))
    )

    # If you enable this line you break parallelism
    # ds.show(limit=1000, truncate_length=500)

    ds.write.opensearch(
        os_client_args=osrch_args,
        index_name=index,
        index_settings=idx_settings,
    )


def import_html(root):
    # TODO: eric - implement HTML import
    print("WARNING, HTML import unimplmented")
    pass


####################################

main()
sys.exit(0)
