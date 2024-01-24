import click

from service.remote_processor_service import RemoteProcessorService

@click.command()
@click.argument('config', type=click.Path(exists=True))
def read_cfg(config):
    """
    Start the server with the configruation frovided

    Args:
        config (filepath): A yaml file that describes all the search processors to run in the RPS
    """
    service = RemoteProcessorService(config)
    print(service)
    print(service._pipelines)
    print(service._pipelines['debug']._processors)