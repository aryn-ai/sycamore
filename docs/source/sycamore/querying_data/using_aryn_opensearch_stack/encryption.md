# Encryption in Sycamore Docker Containers

The Sycamore Docker environment consists of multiple services that communicate over the network.  The most important of these are OpenSearch, Jupyter, and the demo UI.  In an ideal world, all network communication would be encrypted for privacy and security.  The way to do this is via SSL/TLS, which requires that each service be configured with an SSL certificate (X.509 certificate).

There are some real-world complications that affect the Docker Compose environment.  First, the full hostnames of the machines running the containers aren't known in advance.  Instead, the containers use hostnames like `localhost` and aliases provided by Docker.  No trusted certificate authority will issue certificates for `localhost` or `opensearch`.  So, our containers create their own self-signed SSL certificates.

The second issue is that browsers by default don't trust self-signed SSL certificates.  If we can create them, so can any random hacker on the internet.  Each browser is different, but most present a warning upon encountering a self-signed certificate and provide some process to bypass the warning and trust the certificate.

In order to avoid these scary warnings in the demo environment, we decided to disable SSL for Jupyter and the demo query UI.  We still run internal components such as OpenSearch over SSL, but we do not verify the authenticity of certificates, as they are self-signed.

## Forcing SSL Everywhere

If you would like to run Jupyter and the query UI over SSL (HTTPS), it's as easy as setting an environment variable.  Here are two ways to do it:

```
SSL=1 docker compose up --pull=always
```

or

```
export SSL=1
docker compose up --pull=always
```

The `SSL` environment variable does not control the internal services like OpenSearch.  They always run SSL.  For Jupyter and the query UI, we have set the default to `SSL=0` in the `.env` file and each Docker image.  The commands above will override these defaults.
