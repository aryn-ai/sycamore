from argparse import ArgumentParser

from ray.data import ActorPoolStrategy
import sycamore
from sycamore.evaluation.tables.extractors import ExtractTableFromImage
from sycamore.evaluation.tables.table_metrics import TEDSMetric, apply_metric
from sycamore.transforms.table_structure.extract import TableTransformerStructureExtractor

from .benchmark_scans import PubTabNetScan, TableEvalDoc

SCANS = {"pubtabnet": PubTabNetScan}

EXTRACTORS = {"tabletransformer": (TableTransformerStructureExtractor, ActorPoolStrategy(size=2), {"device": "mps"})}

parser = ArgumentParser()
parser.add_argument("dataset", choices=list(SCANS.keys()), help="dataset to evaluate")
parser.add_argument("extractor", choices=list(EXTRACTORS.keys()), help="TableStructureExtractor to evaluate")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
print(args)

metrics = [
    TEDSMetric(structure_only=True),
    TEDSMetric(structure_only=False),
]

ctx = sycamore.init()

sc = SCANS[args.dataset]().to_docset(ctx)
if args.debug:
    sc = sc.limit(10)
extractor, actorpool, kwargs = EXTRACTORS[args.extractor]
extracted = sc.map_batch(ExtractTableFromImage(extractor(**kwargs)), compute=actorpool)
measured = extracted
for m in metrics:
    measured = measured.map(apply_metric(m))

if args.debug:
    doc = measured.take(1)[0]
    ed = TableEvalDoc(doc.data)
    del ed["image"]
    del ed.properties["tokens"]
    print(ed.data)

aggs = measured.plan.execute().aggregate(*[m.to_aggregate_fn() for m in metrics])
print("=" * 80)
print(aggs)
