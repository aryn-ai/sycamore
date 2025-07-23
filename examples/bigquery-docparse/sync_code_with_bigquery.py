#!/usr/bin/env python3
"""
Script to sync stored procedures between local SQL file and BigQuery.

Usage:
    python sync_code_with_bigquery.py [sql_file_path]

    If no sql_file_path is provided, defaults to "stored_procedures.sql"

Example:
    python sync_code_with_bigquery.py
    python sync_code_with_bigquery.py my_procedures.sql

Before running:
    1. Set GOOGLE_PROJECT_ID environment variable: export GOOGLE_PROJECT_ID='your-project-id'
    2. Ensure you have Google Cloud credentials configured
    3. Install required packages: pip install google-cloud-bigquery
"""

import difflib
from google.cloud import bigquery
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional


PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
if not PROJECT_ID:
    print("Error: GOOGLE_PROJECT_ID environment variable is not set")
    print("Please set it with: export GOOGLE_PROJECT_ID='your-project-id'")
    sys.exit(1)

DATASET_ID = os.environ.get("BIGQUERY_DATASET_ID", "example")


def parse_sql_file(file_path: str) -> Dict[str, str]:
    """
    Parse the stored_procedures.sql file and extract individual procedures.

    Args:
        file_path: Path to the SQL file

    Returns:
        Dictionary mapping procedure names to their SQL definitions
    """
    procedures = {}
    print(f"Parsing {file_path}:")
    with open(file_path, "r") as f:
        content = f.read()

    # Split by CREATE OR REPLACE PROCEDURE
    # This regex matches CREATE OR REPLACE PROCEDURE followed by the procedure name and arguments
    # and captures everything until the next CREATE OR REPLACE PROCEDURE or end of file
    pattern = r"CREATE OR REPLACE PROCEDURE\s+([^\s(]+)\s*\(([^)]*)\)\s*BEGIN\n((.*?)\nEND;)"

    matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)

    for match in matches:
        full_name = match.group(1).strip()
        arguments = match.group(2).strip()
        body = match.group(3).rstrip()

        body = "\n".join([line.rstrip() for line in body.split("\n")])

        name = full_name
        name = name.removeprefix(f"`{PROJECT_ID}`.")
        name = name.removeprefix(f"{DATASET_ID}.")

        assert "." not in name, f"Warning: Invalid procedure name '{full_name}'"

        full_procedure = f"CREATE OR REPLACE PROCEDURE `{PROJECT_ID}`.{DATASET_ID}.{name}({arguments})\nBEGIN\n{body}"
        procedures[name] = full_procedure
        print(f"  → {name}")

    return procedures


def get_bigquery_procedure(client: bigquery.Client, procedure_name: str) -> Optional[str]:
    """
    Download a stored procedure from BigQuery.

    Args:
        client: BigQuery client
        procedure_name: Name of the procedure to download

    Returns:
        The procedure definition as a string, or None if not found
    """
    try:
        print("  → Downloading from BigQuery...")
        query = f"""
        SELECT ddl 
        FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.ROUTINES` 
        WHERE routine_name = '{procedure_name}'
        """

        query_job = client.query(query)
        results = list(query_job)

        if results:
            ddl = results[0].ddl.replace("CREATE PROCEDURE", "CREATE OR REPLACE PROCEDURE")
            return ddl
        else:
            return None

    except Exception as e:
        print(f"Error downloading procedure {procedure_name}: {e}")
        return None


def compare_procedures(local_sql: str, bigquery_sql: str) -> str:
    """
    Compare two SQL procedures and return whether they're different.

    Args:
        local_sql: Local procedure definition
        bigquery_sql: BigQuery procedure definition

    Returns:
        Tuple of (is_different, diff_text)
    """
    if local_sql == bigquery_sql:
        return ""

    # Generate diff
    diff = difflib.unified_diff(
        bigquery_sql.splitlines(keepends=True),
        local_sql.splitlines(keepends=True),
        fromfile="BigQuery",
        tofile="Local",
        lineterm="",
    )

    xdiff = [ln if ln.endswith("\n") else ln + "\n" for ln in diff]
    ret = "".join(xdiff)
    assert ret != "", "No differences found but contents were different"
    return ret


def upload_procedure(client: bigquery.Client, procedure_name: str, procedure_sql: str) -> bool:
    """
    Upload a stored procedure to BigQuery.

    Args:
        client: BigQuery client
        procedure_name: Name of the procedure
        procedure_sql: SQL definition of the procedure

    Returns:
        True if successful, False otherwise
    """
    try:
        query_job = client.query(procedure_sql)
        query_job.result()
        print(f"✓ Successfully uploaded procedure: {procedure_name}")
        return True
    except Exception as e:
        print(f"✗ Error uploading procedure {procedure_name}: {e}")
        return False


def find_procedures_to_update(
    client: bigquery.Client, local_procedures: Dict[str, str]
) -> list[tuple[str, str, Optional[str]]]:
    """
    Find procedures that need to be updated by comparing local and BigQuery versions.

    Args:
        client: BigQuery client
        local_procedures: Dictionary of local procedure definitions

    Returns:
        List of tuples (procedure_name, local_sql, bigquery_sql) for procedures that need updating
    """
    procedures_to_update: list[tuple[str, str, Optional[str]]] = []

    for procedure_name, local_sql in local_procedures.items():
        print(f"\nProcessing procedure: {procedure_name}")

        bigquery_sql = get_bigquery_procedure(client, procedure_name)

        if bigquery_sql is None:
            print("  → Procedure not found in BigQuery (will be created)")
            procedures_to_update.append((procedure_name, local_sql, None))
            continue

        diff_text = compare_procedures(local_sql, bigquery_sql)

        if diff_text:
            print_diff(diff_text)
            procedures_to_update.append((procedure_name, local_sql, bigquery_sql))
        else:
            print("  → No differences found")

    return procedures_to_update


def prompt_for_confirmation(items_to_update: List[str], item_type: str = "item") -> bool:
    """
    Prompt user for confirmation before uploading.

    Args:
        items_to_update: List of items to be updated
        item_type: Type of items (e.g., "procedure", "UDF")

    Returns:
        True if user confirms, False if cancelled
    """
    print(f"\n{len(items_to_update)} {item_type}(s) need to be updated:")
    for name in items_to_update:
        print(f"  - {name}")

    print("\n" + "=" * 50)
    print(f"WARNING: This will replace the {item_type}s in BigQuery!")
    print("=" * 50)

    try:
        response = input("\nDo you want to proceed with the upload? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("Operation cancelled by user.")
            return False
        return True
    except KeyboardInterrupt:
        print("\nOperation cancelled by user (Ctrl+C).")
        return False


def sync_stored_procedures(sql_file_path: str) -> None:
    """
    Main function to sync stored procedures between local file and BigQuery.

    Args:
        sql_file_path: Path to the SQL file containing stored procedures
    """
    print(f"Syncing stored procedures from {sql_file_path} to BigQuery...")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print("-" * 50)

    try:
        client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"Error initializing BigQuery client: {e}")
        return

    local_procedures = parse_sql_file(sql_file_path)

    if not local_procedures:
        print("No procedures found in local SQL file.")
        return

    procedures_to_update = find_procedures_to_update(client, local_procedures)

    if not procedures_to_update:
        print("\nAll procedures are up to date!")
        return

    procedure_names = [name for name, _, _ in procedures_to_update]
    if not prompt_for_confirmation(procedure_names, "procedure"):
        return

    print("\nUploading procedures...")
    success_count = 0

    for procedure_name, local_sql, _ in procedures_to_update:
        if upload_procedure(client, procedure_name, local_sql):
            success_count += 1

    print(f"\nUpload complete: {success_count}/{len(procedures_to_update)} procedures updated successfully.")


def get_local_python_as_ddl(function_name: str, sql_params: str, return_type: str, packages: List[str]) -> str:
    paths = [
        f"{function_name}.py",
        f"{function_name}/main.py",
        f"examples/bigquery-docparse/{function_name}.py",
        f"examples/bigquery-docparse/{function_name}/main.py",
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                local_code = f.read()
                local_code_path = path
                break
    else:
        raise ValueError(f"Unable to find code for {function_name} in {paths}")

    config_pattern = r'sys\.path\.append\(os\.path\.dirname\(__file__\) \+ "/\.\."\)\nfrom config import .*\n'
    if re.search(config_pattern, local_code):
        config_dir = Path(local_code_path).parent
        if (config_dir / "config.py").exists():
            config_path = config_dir / "config.py"
        else:
            config_path = config_dir.parent / "config.py"

        assert config_path.exists(), f"Config file {config_path} does not exist"
        with open(config_path, "r") as f:
            config_contents = f.read()
        local_code = re.sub(config_pattern, config_contents, local_code)
    else:
        assert "configs" not in local_code, "configs found in local code w/o proper replacement"

    assert "'''" not in local_code, "need to use ''' to wrap a string so can't use it in the code. use \"\"\" instead"
    local_code = local_code.removesuffix("\n")
    if packages:
        packages_str = ",\n  packages=[\n    " + ",\n    ".join([f'"{pkg}"' for pkg in packages]) + "]"
    else:
        # The vertext-ai-connection bit is how you make connections to external services.
        # I have no idea why Google did it that way.
        packages_str = ""
    return f"""CREATE OR REPLACE FUNCTION `{PROJECT_ID}`.{DATASET_ID}.{function_name}({sql_params}) RETURNS {return_type} LANGUAGE python
WITH CONNECTION `{PROJECT_ID}.us.vertex-ai-connection`
OPTIONS(
  entry_point="{function_name}",
  runtime_version="python-3.11"{packages_str})
AS
r'''
{local_code}
''';"""


def get_bigquery_udf(client: bigquery.Client, function_name: str) -> Optional[str]:
    try:
        print(f"  → Downloading UDF {function_name} from BigQuery")
        query = f"""
        SELECT ddl 
        FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.ROUTINES` 
        WHERE routine_name = '{function_name}'
        """

        query_job = client.query(query)
        results = list(query_job)

        if results:
            ddl = results[0].ddl.replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION")
            return ddl
        else:
            return None

    except Exception as e:
        print(f"Error downloading UDF {function_name}: {e}")
        return None


def print_diff(diff_text: str) -> None:
    print("  → Differences detected:")
    print("    " + "=" * 40)
    for line in diff_text.split("\n"):
        print(f"    {line}")
    print("    " + "=" * 40)


def compare_udf(local_ddl: str, bigquery_ddl: str) -> str:
    if local_ddl.strip() == bigquery_ddl.strip():
        return ""

    diff = difflib.unified_diff(
        bigquery_ddl.splitlines(keepends=True),
        local_ddl.splitlines(keepends=True),
        fromfile="BigQuery",
        tofile="Local",
        lineterm="",
    )

    xdiff = [ln if ln.endswith("\n") else ln + "\n" for ln in diff]
    ret = "".join(xdiff)
    assert ret != "", "No differences found but contents were different"
    return ret


def upload_udf(client: bigquery.Client, function_name: str, ddl: str) -> bool:
    try:
        query_job = client.query(ddl)
        query_job.result()  # Wait for the job to complete
        print(f"✓ Successfully uploaded UDF: {function_name}")
        return True
    except Exception as e:
        print(f"✗ Error uploading UDF {function_name}: {e}")
        return False


def sync_udf(function_name: str, sql_params: str, return_type: str, packages: List[str]) -> None:
    """
    Sync a UDF between local Python file and BigQuery.

    Args:
        function_name: Name of the function
        sql_params: SQL parameter definition
        return_type: SQL return type
        packages: List of Python packages
    """
    print(f"Syncing UDF: {function_name}")

    try:
        client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"Error initializing BigQuery client: {e}")
        return

    local_ddl = get_local_python_as_ddl(function_name, sql_params, return_type, packages)
    bigquery_ddl = get_bigquery_udf(client, function_name)

    if bigquery_ddl is None:
        print("  → UDF not found in BigQuery (will be created)")
        should_upload = True
    else:
        diff_text = compare_udf(local_ddl, bigquery_ddl)

        if diff_text:
            print_diff(diff_text)
            should_upload = True
        else:
            print("  → No differences found")
            should_upload = False

    if not should_upload:
        print("  → UDF is up to date!")
        return

    if not prompt_for_confirmation([function_name], "UDF"):
        return

    print("  → Uploading UDF...")
    if upload_udf(client, function_name, local_ddl):
        print("  → Upload complete!")
    else:
        print("  → Upload failed!")


if __name__ == "__main__":
    sync_stored_procedures("examples/bigquery-docparse/stored_procedures.sql")
    packages = ["functions_framework", "google-cloud-storage", "google-cloud-secret-manager", "aryn-sdk==0.2.8"]
    sync_udf(
        "get_status",
        "async_id STRING",
        "STRUCT<async_id STRING, new_async_id STRING, result STRING, err STRING>",
        packages,
    )
    sync_udf("queue_async", "uri STRING, async_id STRING, config_list STRING", "STRING", packages)
    sync_udf("sleep_until", "unix_timestamp INT64", "STRING", [])
