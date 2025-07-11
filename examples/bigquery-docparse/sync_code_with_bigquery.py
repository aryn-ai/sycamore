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

import re
import sys
import os
from typing import Dict, List, Tuple, Optional
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import difflib


PROJECT_ID = os.environ.get('GOOGLE_PROJECT_ID')
if not PROJECT_ID:
    print("Error: GOOGLE_PROJECT_ID environment variable is not set")
    print("Please set it with: export GOOGLE_PROJECT_ID='your-project-id'")
    sys.exit(1)

DATASET_ID = os.environ.get('BIGQUERY_DATASET_ID', "example")


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
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by CREATE OR REPLACE PROCEDURE
    # This regex matches CREATE OR REPLACE PROCEDURE followed by the procedure name and arguments
    # and captures everything until the next CREATE OR REPLACE PROCEDURE or end of file
    pattern = r'CREATE OR REPLACE PROCEDURE\s+([^\s(]+)\s*\(([^)]*)\)\s*BEGIN\n((.*?)\nEND;)'
    
    matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        full_name = match.group(1).strip()
        arguments = match.group(2).strip()
        body = match.group(3).rstrip()
        
        body_lines = '\n'.join([line.rstrip() + "\n" for line in body.split('\n')])
        
        name = full_name
        name = name.removeprefix(f"`{PROJECT_ID}`.")
        name = name.removeprefix(f"{DATASET_ID}.")
        
        assert '.' not in name, f"Warning: Invalid procedure name '{full_name}'"
        
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
            #print(f"ERICDDL\n-----------------------------\n{ddl}\n----------------------------")
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
        fromfile='BigQuery',
        tofile='Local',
        lineterm=''
    )
    #print(f"ERICDiff\n-----------------------------\n{bigquery_sql}\n------------------------------\n{local_sql}\n-----------------------------\n{diff}----------------------------")

    xdiff = [l if l.endswith('\n') else l + '\n' for l in diff]
    #print(f'ERICXDIFF\n-----------------------------\n{xdiff}\n----------------------------')
    ret = ''.join(xdiff)
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


def find_procedures_to_update(client: bigquery.Client, local_procedures: Dict[str, str]) -> List[Tuple[str, str, Optional[str]]]:
    """
    Find procedures that need to be updated by comparing local and BigQuery versions.
    
    Args:
        client: BigQuery client
        local_procedures: Dictionary of local procedure definitions
        
    Returns:
        List of tuples (procedure_name, local_sql, bigquery_sql) for procedures that need updating
    """
    procedures_to_update = []
    
    for procedure_name, local_sql in local_procedures.items():
        print(f"\nProcessing procedure: {procedure_name}")
        
        bigquery_sql = get_bigquery_procedure(client, procedure_name)
        
        if bigquery_sql is None:
            print(f"  → Procedure not found in BigQuery (will be created)")
            procedures_to_update.append((procedure_name, local_sql, None))
            continue
        
        diff_text = compare_procedures(local_sql, bigquery_sql)
        #print(f"ERICDIFF\n-----------------------------\n{diff_text}\n----------------------------")
        
        if diff_text:
            print(f"  → Differences detected:")
            print("    " + "=" * 40)
            for line in diff_text.split('\n'):
                print(f"    {line}")
            print("    " + "=" * 40)
            procedures_to_update.append((procedure_name, local_sql, bigquery_sql))
        else:
            print(f"  → No differences found")
    
    return procedures_to_update


def prompt_for_confirmation(procedures_to_update: List[Tuple[str, str, Optional[str]]]) -> bool:
    """
    Prompt user for confirmation before uploading procedures.
    
    Args:
        procedures_to_update: List of procedures to be updated
        
    Returns:
        True if user confirms, False if cancelled
    """
    print(f"\n{len(procedures_to_update)} procedure(s) need to be updated:")
    for name, _, _ in procedures_to_update:
        print(f"  - {name}")
        
    try:
        response = input("\nDo you want to proceed with the upload? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
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
    
    if not prompt_for_confirmation(procedures_to_update):
        return
    
    print("\nUploading procedures...")
    success_count = 0
    
    for procedure_name, local_sql, _ in procedures_to_update:
        if upload_procedure(client, procedure_name, local_sql):
            success_count += 1
    
    print(f"\nUpload complete: {success_count}/{len(procedures_to_update)} procedures updated successfully.")


if __name__ == "__main__":
    sql_file = "examples/bigquery-docparse/stored_procedures.sql"
    sync_stored_procedures(sql_file)
