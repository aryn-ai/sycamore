from argparse import ArgumentParser

from ray.data import ActorPoolStrategy
import sycamore
from sycamore.context import ExecMode
from sycamore.evaluation.ocr.models import PaddleOCR, EasyOCR, Tesseract, LegacyOCR, ExtractOCRFromImage
from sycamore.evaluation.ocr.metrics import CharacterErrorRate, WordErrorRate, MatchErrorRate, apply_metric

from sycamore.evaluation.ocr.data import BaseOCREvalScan, HandwritingOCREvalScan, InvoiceOCREvalScan

DATASETS = {"base": BaseOCREvalScan, "handwriting": HandwritingOCREvalScan, "invoice": InvoiceOCREvalScan}

MODELS = {"paddle": PaddleOCR, "easy": EasyOCR, "tesseract": Tesseract, "legacy": LegacyOCR}

METRICS = [CharacterErrorRate(), MatchErrorRate(), WordErrorRate()]

model_actorpool = ActorPoolStrategy(size=2)
model_kwargs = {"device": "mps"}

parser = ArgumentParser()
parser.add_argument("dataset", required=False, choices=list(DATASETS.keys()), help="dataset to evaluate")
parser.add_argument("model", required=False, choices=list(MODELS.keys()), help="OCR Model to use")
parser.add_argument("--debug", required=False, action="store_true")
parser.add_argument("--limit", type=int, default=10000, help="A limit on the number of values to run")
args = parser.parse_args()
dataset = DATASETS.get(args.dataset, BaseOCREvalScan) if args.dataset else BaseOCREvalScan
model = MODELS.get(args.model, EasyOCR) if args.model else EasyOCR
# debug = args.debug if args.debug else False
limit = args.limit if not args.debug else args.debug

# ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
ctx = sycamore.init()
sc = dataset().to_docset(ctx)  # type: ignore
sc = sc.limit(limit)
measured = sc.map_batch(ExtractOCRFromImage(model()), compute=model_actorpool)
for m in METRICS:
    measured = measured.map(apply_metric(m))

# if debug:
#     doc = measured.take(1)[0]
#     ed = OCREvalDocument(doc.data)
#     del ed["image"]
#     # del ed.gt_text if "gt_text" in ed
#     print(ed.data)

aggs = measured.plan.execute().aggregate(*[m.to_aggregate_fn() for m in METRICS])
print("=" * 80)
print(aggs)
