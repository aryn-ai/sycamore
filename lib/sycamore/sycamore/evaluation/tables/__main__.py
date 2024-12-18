from argparse import ArgumentParser
from tqdm import tqdm
from ray.data import ActorPoolStrategy
import sycamore
from sycamore.context import ExecMode
from sycamore.evaluation.tables.extractors import ExtractTableFromImage, FlorenceTableStructureExtractor, HomemadeTableTransformerTableStructureExtractor, PaddleTableStructureExtractor, TextractTableStructureExtractor, PaddleV2TableStructureExtractor
from sycamore.evaluation.tables.table_metrics import TEDSMetric, apply_metric
from sycamore.transforms.table_structure.extract import TableTransformerStructureExtractor

from .benchmark_scans import CohereTabNetS3Scan, FinTabNetS3Scan, PubTabNetScan, TableEvalDoc
from time import time

SCANS = {"pubtabnet": PubTabNetScan, "fintabnet": FinTabNetS3Scan, "coheretabnet": CohereTabNetS3Scan}

EXTRACTORS = {
    "tabletransformer": (TableTransformerStructureExtractor, ActorPoolStrategy(size=1), {"device": "cuda:0"}),
    "paddleocr": (PaddleTableStructureExtractor, None, {}),
    "paddlev2": (PaddleV2TableStructureExtractor, None, {}),
    "textract": (TextractTableStructureExtractor, None, {}),
    "florence": (FlorenceTableStructureExtractor, None, {}),
    "homemade": (HomemadeTableTransformerTableStructureExtractor, ActorPoolStrategy(size=1), {"device": "cuda:0"}),
}


def local_aggregate(docs, *agg_fns):
    aggcumulations = {af.name: af.init(af.name) for af in agg_fns}
    for doc in docs:
        for af in agg_fns:
            aggcumulations[af.name] = af.accumulate_row(aggcumulations[af.name], doc, in_ray=False)
    return {af.name: af.finalize(aggcumulations[af.name]) for af in agg_fns}


parser = ArgumentParser()
parser.add_argument("dataset", choices=list(SCANS.keys()), help="dataset to evaluate")
parser.add_argument("extractor", choices=list(EXTRACTORS.keys()), help="TableStructureExtractor to evaluate")
parser.add_argument("--debug", action="store_true")
parser.add_argument("-l", "--limit", default=-1, type=int, required=False)
parser.add_argument("--noreal", action="store_true")
args = parser.parse_args()
print(args)

metrics = [
    TEDSMetric(structure_only=True),
    # TEDSMetric(structure_only=False),
]

local_ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
# ray_ctx = sycamore.init()

# sc = SCANS[args.dataset]().to_docset(ray_ctx)
# if args.debug:
#     sc = sc.limit(10)

docs = []
docgenerator = iter(SCANS[args.dataset]().local_process(limit=args.limit))
if args.debug:
    if args.limit == -1:
        args.limit = 10
    for _ in tqdm(range(args.limit), desc="Loading documents"):
        try:
            docs.append(next(docgenerator))
        except StopIteration:
            break
else:
    for doc in tqdm(docgenerator, desc="Loading documents"):
        docs.append(doc)
print(f"Loaded {len(docs)} documents")

start = time()
extractor, actorpool, kwargs = EXTRACTORS[args.extractor]
extracted = local_ctx.read.document(docs).map_batch(ExtractTableFromImage(extractor(**kwargs)))
measured = extracted
for m in metrics:
    measured = measured.map(apply_metric(m))

if args.debug:
    doc = measured.take(2)[1]
    ed = TableEvalDoc(doc.data)
    del ed["image"]
    del ed.properties["tokens"]
    print(ed.gt_table.to_html())
    print(ed.pred_table.to_html())
    print(ed.data)

if args.noreal:
    exit()
# aggs = measured.plan.execute().aggregate(*[m.to_aggregate_fn() for m in metrics])
aggs = local_aggregate(measured.take_all(), *[m.to_aggregate_fn(in_ray=False) for m in metrics])
print("=" * 80)
print(aggs)
print(f"Time spent for extraction and metrics calculation: {time() - start} seconds.")
