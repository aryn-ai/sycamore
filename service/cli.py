import click

from service.remote_processor_service import RemoteProcessorService

@click.command()
@click.argument('config', type=click.Path(exists=True))
def read_cfg(config):
    """
    Construct the server with the configruation frovided and print info about it

    Args:
        config (filepath): A yaml file that describes all the search processors to run in the RPS
    """
    service = RemoteProcessorService(config)
    print(service)
    print(service._pipelines)
    print(service._pipelines['debug']._processors)

@click.command()
@click.argument('config', type=click.Path(exists=True))
def serve(config):
    """
    Start the server on port 2796 with the configuration provided

    Args:
        config (filepath): A yaml file that describes all the search processors to run in the RPS
    """
    service = RemoteProcessorService(config)
    server = service.start()
    server.wait_for_termination()