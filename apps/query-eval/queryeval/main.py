#!/usr/bin/env python

# This is the CLI for the Sycamore Query Evaluation app, which benchmarks the
# performance of Sycamore Query for generating query plans and/or answering analytical questions
# from a dataset.
#
# Usage:
#   poetry run python queryeval/main.py \
#      --outfile results.yaml \
#      --index const_ntsb \
#      data/ntsb-queries.yaml \
#      run

from typing import Optional

import click
from rich.console import Console


from queryeval.driver import QueryEvalDriver


console = Console()


@click.group()
@click.argument("config-file", type=click.Path(exists=True))
@click.option("--index", help="OpenSearch index name")
@click.option("--outfile", help="Output file", required=True)
@click.option("--logfile", help="Detailed log file")
@click.option("--query-cache-path", help="Query cache path")
@click.option("--llm-cache-path", help="LLM cache path")
@click.option("--dry-run", help="Dry run - do not run any stages", is_flag=True)
@click.option("--doc-limit", help="Limit number of docs in result set", type=int)
@click.option("--overwrite", help="Overwrite existing results file", is_flag=True)
@click.option(
    "--raw-output", help="Output should be a raw DocSet, rather than natural language", is_flag=True, default=False
)
@click.pass_context
def cli(
    ctx,
    config_file: str,
    index: str,
    outfile: str,
    logfile: str,
    query_cache_path: Optional[str],
    llm_cache_path: Optional[str],
    dry_run: bool,
    doc_limit: Optional[int],
    overwrite: bool,
    raw_output: bool,
):
    ctx.ensure_object(dict)
    driver = QueryEvalDriver(
        input_file_path=config_file,
        index=index,
        results_file=outfile,
        query_cache_path=query_cache_path,
        llm_cache_path=llm_cache_path,
        dry_run=dry_run,
        natural_language_response=not raw_output,
        log_file=logfile,
        doc_limit=doc_limit,
        overwrite=overwrite,
    )
    ctx.obj["driver"] = driver


@cli.command()
@click.pass_context
def run(
    ctx,
):
    """Run all query eval stages: plan, query, and eval."""
    driver = ctx.obj["driver"]
    driver.run()


@cli.command()
@click.pass_context
def plan(
    ctx,
):
    """Run the plan stage."""
    driver = ctx.obj["driver"]
    driver.plan_all()


@cli.command()
@click.pass_context
def query(
    ctx,
):
    """Run the query stage."""
    driver = ctx.obj["driver"]
    driver.query_all()


@cli.command()
@click.pass_context
def eval(  # pylint: disable=redefined-builtin
    ctx,
):
    """Run the eval stage."""
    driver = ctx.obj["driver"]
    driver.eval_all()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
