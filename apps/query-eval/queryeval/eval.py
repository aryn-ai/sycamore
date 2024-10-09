#!/usr/bin/env python

# This script reads in a previous result of running lunabench.py
# and annotates each result with a set of evaluation metrics.

from typing import List

import click
from datasets import Dataset
from pydantic_yaml import to_yaml_str
from ragas import evaluate
from ragas.metrics import answer_relevancy, answer_correctness, answer_similarity
from rich.console import Console
from yaml import safe_load_all

from lunabench import LunaBenchEvaluation, LunaBenchResult, LunaBenchResults


console = Console()


class LunaBenchEval:
    """Class to run LunaBench evaluations.

    Args:
        results_file_path: Path to the benchmark results file.
    """

    def __init__(
        self,
        results_file_path: str,
    ):
        self.results_file_path = results_file_path

    def read_results_file(self) -> List[LunaBenchResults]:
        results_list: List[LunaBenchResults] = []
        with open(self.results_file_path, "r", encoding="utf8") as f:
            docs = safe_load_all(f)
            for doc in docs:
                if doc is None:
                    continue
                result = LunaBenchResults(**doc)
                results_list.append(result)
        return results_list

    def write_results_file(self, results_list: List[LunaBenchResults]):
        with open(self.results_file_path, "w", encoding="utf8") as f:
            for results in results_list:
                f.write(to_yaml_str(results))
                f.write("---\n")

    def evaluate_results(self, results: LunaBenchResults, force: bool = False) -> LunaBenchResults:
        for result in results.results:
            if result.evaluation and not force:
                console.print(f"Skipping evaluation for [green]{result.query.query}")
                continue
            result.evaluation = self.evaluate_result(result)
        return results

    def evaluate_result(self, result: LunaBenchResult) -> LunaBenchEvaluation:
        data_samples = {
            "question": [result.query.query],
            "answer": [result.response],
            "ground_truth": [result.query.expected],
        }
        dataset = Dataset.from_dict(data_samples)
        ragas_result = evaluate(
            dataset,
            metrics=[
                answer_correctness,
                answer_similarity,
            ],
        )
        for _, row in ragas_result.to_pandas().iterrows():
            return LunaBenchEvaluation(
                correctness_score=row["answer_correctness"],
                similarity_score=row["answer_similarity"],
            )


@click.group()
@click.argument("results-file", type=click.Path(exists=True))
@click.pass_context
def cli(ctx, results_file: str):
    ctx.obj = LunaBenchEval(results_file_path=results_file)


@cli.command()
@click.option("--force", help="Re-evaluate if evaluation already exists", is_flag=True)
@click.pass_context
def eval(ctx, force: bool):
    results_list = ctx.obj.read_results_file()
    new_results_list = []
    for results in results_list:
        results = ctx.obj.evaluate_results(results, force=force)
        new_results_list.append(results)
        for result in results.results:
            console.print(result.evaluation)
    ctx.obj.write_results_file(new_results_list)
    console.print(f"Results written to {ctx.obj.results_file_path}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
