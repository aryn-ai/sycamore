import pytest
from pyarrow.fs import LocalFileSystem

from sycamore import ExecMode
from sycamore.data.document import Document


# ============================================================================
# ML Model Fakes for Fast Unit Tests
# ============================================================================


@pytest.fixture
def fake_detr():
    """
    Provides a FakeDeformableDetr instance that returns pre-recorded ground truth.

    Use this when you need explicit control over the fake DETR model.
    For automatic injection into all unit tests, use the use_fake_detr fixture instead.
    """
    from sycamore.tests.unit.transforms.partitioner_fakes import FakeDeformableDetr

    return FakeDeformableDetr()


@pytest.fixture
def fake_table_extractor():
    """
    Provides a FakeTableStructureExtractor instance that returns pre-recorded ground truth.

    Use this when you need explicit control over the fake table extractor.
    """
    from sycamore.tests.unit.transforms.partitioner_fakes import FakeTableStructureExtractor

    return FakeTableStructureExtractor()


@pytest.fixture
def use_fake_detr(monkeypatch, fake_detr):
    """
    Patches DeformableDetr to use the fake implementation.

    When this fixture is used, any code that instantiates DeformableDetr will
    get the fake instead, returning pre-recorded ground truth data.

    Example:
        def test_partition(use_fake_detr):
            # DeformableDetr will return ground truth instead of running inference
            partitioner = ArynPDFPartitioner()
            ...
    """
    from sycamore.tests.unit.transforms.partitioner_fakes import FakeDeformableDetr

    monkeypatch.setattr(
        "sycamore.transforms.detr_partitioner.DeformableDetr",
        FakeDeformableDetr,
    )
    return fake_detr


@pytest.fixture
def use_fake_table_extractor(monkeypatch, fake_table_extractor):
    """
    Patches the default table structure extractor to use the fake implementation.

    Example:
        def test_table_extraction(use_fake_table_extractor):
            # Table extraction will return ground truth instead of running inference
            ...
    """
    from sycamore.tests.unit.transforms.partitioner_fakes import FakeTableStructureExtractor

    monkeypatch.setattr(
        "sycamore.transforms.table_structure.extract.DEFAULT_TABLE_STRUCTURE_EXTRACTOR",
        lambda **kwargs: FakeTableStructureExtractor(**kwargs),
    )
    return fake_table_extractor


@pytest.fixture
def use_fake_models(use_fake_detr, use_fake_table_extractor):
    """
    Patches both DeformableDetr and table structure extractor to use fake implementations.

    This is a convenience fixture that combines use_fake_detr and use_fake_table_extractor.

    Example:
        def test_full_partition(use_fake_models):
            # Both DETR and table extraction will use ground truth
            partitioner = ArynPDFPartitioner()
            ...
    """
    return use_fake_detr, use_fake_table_extractor


# ============================================================================
# Standard Fixtures
# ============================================================================


@pytest.fixture
def read_local_binary(request) -> Document:
    local = LocalFileSystem()
    path = str(request.param)
    input_stream = local.open_input_stream(path)
    document = Document()
    document.binary_representation = input_stream.readall()
    document.properties["path"] = path
    return document


@pytest.fixture(params=(exec_mode for exec_mode in ExecMode if exec_mode != ExecMode.UNKNOWN))
def exec_mode(request):
    """
    Use this to run a test against all available execution modes. You will need to pass this as a parameter to
    the Context initialization. e.g.

    Example:
        .. code-block:: python

            def test_example(exec_mode):
                context = sycamore.init(exec_mode=exec_mode)
                ...
    """
    return request.param
