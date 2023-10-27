from pathlib import Path
import os
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

# TODO: eric - detect that the opensearch cluster has dropped documents and reload them
# TODO: eric - figure out how to deal with documents that are deleted
# TODO: eric - adjust the way we do importing so that the files have a permanent name so the viewPdf UI option works
def main():
    root_path = '/app/.scrapy'
    if len(sys.argv) <= 1:
        if not os.path.isdir(root_path):
            raise RuntimeError('Missing ' + root_path + '; run with single path argument if not in container')
    elif len(sys.argv) != 2:
        raise RuntimeError('Usage: docker_local_injest.py [directory_root]')
    else:
        root_path = sys.argv[1]
        if not os.path.isdir(root_path):
            raise RuntimeError('Missing specified path ' + root_path + '; correct argument or remove if in container')

    if root_path == '/app/.scrapy':
        print('Assuming execution is in container, using adjusted host')
        osrch_args['hosts'][0]['host'] = 'aryn_opensearch'

    if root_path == '/app/.scrapy' and 'OPENAI_API_KEY' in os.environ:
        print ('WARNING: OPENAI_API_KEY in environment was probably passed insecurely into docker.')
        # TODO: eric - switch over to docker swarm so we can use docker secrets
        # Then enable the sleep since we shouldn't get the env var any more.
        #print "sleep(300)"
        #time.sleep(300)

    if 'OPENAI_API_KEY' not in os.environ:
        raise RuntimeError('Missing OPENAI_API_KEY')
    
    root = Path(root_path)

    while True:
        files = find_files(root)
        print('Files:', files)
        if len(files) > 0:
            import_files(root, files)
            time.sleep(1)
        else:
            print('No changes, sleeping')
            time.sleep(5)

def find_files(root):
    downloads = root.joinpath('downloads')
    pdf_files = find_recursive(downloads, 'pdf')
    html_files = find_recursive(downloads, 'html')
    ret = []
    for i in pdf_files + html_files:
        t = reimport_timestamp(root, i['suffix'])
        if t > 0:
            print('DEBUG1 reimport', i['suffix'])
            i['timestamp'] = t
            ret.append(i)
        else:
            print('DEBUG1 unchanged', i['suffix'])

    return ret
    
def find_recursive(root, file_type):
    prefix = Path(root)
    path = os.path.join(prefix, file_type)
    ret = []
    for path in Path(root).rglob('*'):
        ret.append({'path': path, 'suffix': path.relative_to(prefix), 'type': file_type})

    return ret

def import_files(root, files):
    pending_paths = {}
    for i in files:
        suffix = i['suffix']
        pending = root.joinpath('pending', suffix)
        os.makedirs(os.path.dirname(pending), exist_ok=True)
        if os.path.isfile(pending):
            pending.unlink()
        os.link(i['path'], pending)
        if not i['type'] in pending_paths:
            pending_paths[i['type']] = []

        pending_paths[i['type']].append(str(pending))

    if 'pdf' in pending_paths:
        import_PDF(pending_paths['pdf'])

    if 'html' in pending_paths:
        import_HTML(pending_paths['html'])
    
    for i in files:
       imported = root.joinpath('imported', i['suffix'])
       os.makedirs(os.path.dirname(imported), exist_ok=True)
       if not os.path.isfile(imported):
           imported.touch()
       os.utime(imported, (time.time(), i['timestamp']))

def reimport_timestamp(root, suffix):
    s = os.lstat(root.joinpath('downloads', suffix))
    if not stat.S_ISREG(s.st_mode):
        print('DEBUG3 download notfile', suffix)
        return -1
    
    file_timestamp = s.st_mtime
    
    try:
        s = os.lstat(root.joinpath('imported', suffix))
    except FileNotFoundError:
        print('DEBUG4 missing')
        return file_timestamp

    if s.st_mtime < file_timestamp:
        print('DEBUG5 newer')
        return file_timestamp

    if s.st_mtime > file_timestamp:
        print('WARNING file got older', root, suffix, file_timestamp, s.st_mtime)
        return file_timestamp

    print('DEBUG6 unchanged')
    return 0

def import_PDF(paths):
    index = 'demoindex0'

    davinci_llm = OpenAI(OpenAIModels.TEXT_DAVINCI.value)
    
    ctx = sycamore.init()
    ds = (
        ctx.read.binary(paths, binary_format='pdf')
        .partition(partitioner=UnstructuredPdfPartitioner())
        .extract_entity(entity_extractor=OpenAIEntityExtractor('title', llm=davinci_llm, prompt_template=title_template))
        .spread_properties(['path', 'title'])
        .explode()
        .embed(embedder=SentenceTransformerEmbedder(model_name='all-MiniLM-L6-v2', batch_size=100))
    )

    # If you enable this line you break parallelism
    # ds.show(limit=1000, truncate_length=500)

    ds.write.opensearch(
        os_client_args=osrch_args,
        index_name=index,
        index_settings=idx_settings,
    )

def import_HTML(root):
    pass

####################################3

main()
sys.exit(0)
