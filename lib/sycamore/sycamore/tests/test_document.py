from sycamore.document import Caption, NaiveSemanticVisitor, Table


def test_caption_accept():
    node_id = 1
    metadata = {"author": "John Doe"}
    caption = Caption(node_id, [1, 2, 3, 4], "Caption content", metadata)
    visitor = NaiveSemanticVisitor()

    semantic = caption.accept(visitor)

    # Assert
    assert semantic == "Caption content"


def test_table_accept():
    from sycamore.data import Table as TableContent, TableCell

    node_id = 1
    table = TableContent(
        [
            TableCell(content="head1", rows=[0], cols=[0], is_header=True),
            TableCell(content="head2", rows=[0], cols=[1], is_header=True),
            TableCell(content="3", rows=[1], cols=[0], is_header=False),
            TableCell(content="4", rows=[1], cols=[1], is_header=False),
        ]
    )
    metadata = {"author": "John Doe"}
    table = Table(node_id, [1, 2, 3, 4], table, metadata)
    visitor = NaiveSemanticVisitor()

    semantic = table.accept(visitor)

    assert semantic == "head1,head2\n3,4\n"
