# Neo4j

[Neo4j](https://neo4j.com/) is a full-featured, open-source, native graph database and analytics engine.

## Configuration for Neo4j

*Please see Neo4j's [installation](https://neo4j.com/docs/operations-manual/current/docker/introduction/) page for more in-depth information on installing, configuring, and running Neo4j.*

We recommend running Neo4j through docker run. The following APOC environment variables are required for sycamore to write to Neo4j.

```bash
docker run -d \
-p 7474:7474 -p 7687:7687 \
--name neo4j \
--env NEO4J_apoc_export_file_enabled=true \
--env NEO4J_apoc_import_file_enabled=true \
--env NEO4J_PLUGINS=\[\"apoc\"\] \
--env NEO4J_AUTH=neo4j/{REPLACE-WITH-PASSWORD} \
--volume {REPLACE-WITH-TEMP-PATH}:/import \
neo4j:latest
```

Note: You may have to run `sudo chmod -R 777 {REPLACE-WITH-TEMP-PATH}` to give read and write access to neo4j's import directory.

## Writing to Neo4j

To write a DocSet to a Neo4j instance from Sycamore, use the `docset.write.neo4j(...)` function. The Neo4j writer takes the following arguments:

- `uri`: Connection endpoint for the neo4j instance. Note that this must be paired with the
    necessary client arguments below
- `auth`: Authentication arguments to be specified. See more information at
    https://neo4j.com/docs/api/python-driver/current/api.html#auth-ref
- `import_dir`: The import directory that neo4j uses. You can specify where to mount this volume when you launch
    your neo4j docker container.
- `database`: (Optional, default=`neo4j`) Database to write to in Neo4j. By default in the neo4j community addition, new databases
    cannot be instantiated so you must use "neo4j". If using enterprise edition, ensure the database exists.
- `use_auradb`: (Optional, default=`False`) Set to true if you are using neo4j's serverless implementation called AuraDB.
- `s3_session`: (Optional, default=`None`) An AWS S3 Session. This is used as a proxy to securly upload your files into AuraDB. Defaults to None. This field is required if use_auradb is set to true.


To use the Neo4j Writer, you must follow the   
```python
ds = (
    ctx.read.binary(...),
    .partition(...),
    .extract_document_structure(...),
    .extract_graph_entities(...),
    .extract_graph_relationships(...),
    .resolve_graph_entities(...),
    .explode()
)

URI = "neo4j+s://<AURADB_INSTANCE_ID>.databases.neo4j.io"
AUTH = ("neo4j", "sample_password")
DATABASE = "neo4j
IMPORT_DIR = "/tmp/neo4j"
S3_SESSION = boto3.session.Session()

ds.write.neo4j(
    uri=URI, 
    auth=AUTH, 
    database=DATABASE, 
    import_dir=IMPORT_DIR, 
    use_auradb=True, 
    s3_session=S3_SESSION
)
```

More information can be found in the {doc}`API documentation <../APIs/docsetwriter>`.
