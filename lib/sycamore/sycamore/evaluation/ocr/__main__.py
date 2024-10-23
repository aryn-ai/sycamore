from argparse import ArgumentParser

from ray.data import ActorPoolStrategy
import sycamore
from sycamore.evaluation.ocr.models import (
    PaddleOCR,
    EasyOCR,
    Tesseract,
    Textract,
    LLMOCR,
    DocTR,
    RapidOCR,
    ExtractOCRFromImage,
)
from sycamore.evaluation.ocr.metrics import (
    CharacterErrorRate,
    WordErrorRate,
    MatchErrorRate,
    WordInformationLost,
    apply_metric,
)
import time
import json
from sycamore.evaluation.ocr.data import BaseOCREvalScan, InvoiceOCREvalScan, HandwritingOCREvalScan

DATASETS = {"base": BaseOCREvalScan, "handwriting": HandwritingOCREvalScan, "invoice": InvoiceOCREvalScan}

MODELS = {
    "easy": EasyOCR,
    "tesseract": Tesseract,
    "textract": Textract,
    "paddle": PaddleOCR,
    "llm": LLMOCR,
    "doctr": DocTR,
    "rapid": RapidOCR,
}

METRICS = [CharacterErrorRate(), MatchErrorRate(), WordErrorRate(), WordInformationLost()]

model_actorpool = ActorPoolStrategy(size=2)
model_kwargs = {"device": "mps"}

parser = ArgumentParser()
parser.add_argument("dataset", nargs="?", choices=list(DATASETS.keys()), help="dataset to evaluate")
parser.add_argument("model", nargs="?", choices=list(MODELS.keys()), help="OCR Model to use")
parser.add_argument("--debug", required=False, action="store_true")
parser.add_argument("--limit", type=int, default=1000, help="A limit on the number of values to run")
args = parser.parse_args()
dataset = DATASETS.get(args.dataset, BaseOCREvalScan) if args.dataset else BaseOCREvalScan
model = MODELS.get(args.model, EasyOCR) if args.model else EasyOCR
# debug = args.debug if args.debug else False
limit = args.limit if not args.debug else args.debug

all_results: dict[str, dict[str, dict[str, float]]] = {}
for dataset_name, dataset_class in DATASETS.items():
    ctx = sycamore.init()

    pipeline = dataset_class().to_docset(ctx)  # type: ignore
    if dataset_name == "base":
        pipeline = pipeline.filter(lambda doc: "index" not in doc.data)  # type: ignore
    all_results[dataset_name] = {}
    for model_name, model_class in MODELS.items():
        print(f"Running evaluation for dataset '{dataset_name}' and model '{model_name}'")

        curr_time = time.time()
        pipeline = pipeline.limit(limit)
        pipeline = pipeline.map_batch(ExtractOCRFromImage(model_class()), compute=model_actorpool)
        for m in METRICS:
            pipeline = pipeline.map(apply_metric(m))
        aggs = pipeline.plan.execute().aggregate(*[m.to_aggregate_fn() for m in METRICS])
        aggs["latency"] = time.time() - curr_time

        print("=" * 80)
        print(aggs)
        all_results[dataset_name][model_name] = aggs

        # Write the results to a JSON file
        with open("ocr_evaluation_results.json", "w") as f:
            json.dump(all_results, f, indent=4)
