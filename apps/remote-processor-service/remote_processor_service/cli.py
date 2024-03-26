import click

from remote_processors.server.remote_processor_service import RemoteProcessorService


@click.command()
@click.argument("config", type=click.Path(exists=True))
def read_cfg(config):
    """
    Construct the server with the configruation frovided and print info about it

    Args:
        config (filepath): A yaml file that describes all the search processors to run in the RPS
    """
    service = RemoteProcessorService(config)
    print(service)
    print(service._pipelines)


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--certfile", type=click.Path(exists=True), default=None)
@click.option("--keyfile", type=click.Path(exists=True), default=None)
def serve(config, certfile, keyfile):
    """
    Start the server on port 2796 with the configuration provided

    Args:
        config (filepath): A yaml file that describes all the search processors to run in the RPS
    """
    service = RemoteProcessorService(config)
    if certfile is None or keyfile is None:
        assert keyfile == certfile, "You must either specify both certfile and keyfile or specify neither"
    server = service.start(certfile, keyfile)
    server.wait_for_termination()
