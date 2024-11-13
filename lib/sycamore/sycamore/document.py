from abc import abstractmethod, ABC
from enum import Enum


class Category(Enum):
    Caption = 1
    Footnote = 2
    Formula = 3
    ListItem = 4
    PageFooter = 5
    PageHeader = 6
    Picture = 7
    SectionHeader = 8
    Table = 9
    Text = 10
    Title = 11
    Section = 101
    Group = 102


class Node:
    def __init__(self, node_id, properties):
        self._node_id = node_id
        self._properties = properties

    def node_id(self):
        return self._node_id

    @abstractmethod
    def category(self):
        pass

    def properties(self):
        return self._properties

    @abstractmethod
    def accept(self, visitor):
        pass


class Leaf(Node):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, properties)
        self._box = box
        self._content = content

    def box(self):
        return self._box

    def content(self):
        return self._content


class Internal(Node):
    def __init__(self, node_id, children, properties):
        super().__init__(node_id, properties)
        self._children = children

    def children(self) -> list[Node]:
        return self._children


class Caption(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_caption(self)


class Footnote(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_footnote(self)


class Formula(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_formula(self)


class ListItem(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_list_item(self)


class PageFooter(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_page_footer(self)


class PageHeader(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_page_header(self)


class Picture(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_picture(self)


class SectionHeader(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_section_header(self)


class Table(Leaf):
    def __init__(self, node_id, box, content, properties, continued=None):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_table(self)


class Text(Leaf):
    def __init__(self, node_id, box, content, properties, continued=None):
        super().__init__(node_id, box, content, properties)
        self._continued = continued

    def category(self):
        return Category.Caption

    def continued(self):
        return self._continued

    def accept(self, visitor):
        return visitor.visit_text(self)


class Title(Leaf):
    def __init__(self, node_id, box, content, properties):
        super().__init__(node_id, box, content, properties)

    def category(self):
        return Category.Caption

    def accept(self, visitor):
        return visitor.visit_title(self)


class Group(Internal):
    """
    Group semantic related objects together, e.g. list items, figure and caption
    """

    def __init__(self, node_id, children, properties):
        super().__init__(node_id, children, properties)

    def category(self):
        return Category.Group

    def accept(self, visitor):
        return visitor.visit_group(self)


class Section(Internal):
    def __init__(self, node_id, children, header, properties):
        super().__init__(node_id, children, properties)
        self._header = header

    def category(self):
        return Category.Section

    def accept(self, visitor):
        return visitor.visit_section(self)


class Visitor(ABC):
    @abstractmethod
    def visit_caption(self, caption: Caption):
        pass

    @abstractmethod
    def visit_footnote(self, footnote: Footnote):
        pass

    @abstractmethod
    def visit_formula(self, formula: Formula):
        pass

    @abstractmethod
    def visit_list_item(self, list_item: ListItem):
        pass

    @abstractmethod
    def visit_page_footer(self, page_footer: PageFooter):
        pass

    @abstractmethod
    def visit_page_header(self, page_header: PageHeader):
        pass

    @abstractmethod
    def visit_picture(self, picture: Picture):
        pass

    @abstractmethod
    def visit_section_header(self, section_header: SectionHeader):
        pass

    @abstractmethod
    def visit_table(self, table: Table):
        pass

    @abstractmethod
    def visit_text(self, text: Text):
        pass

    @abstractmethod
    def visit_title(self, title: Title):
        pass

    @abstractmethod
    def visit_group(self, group: Group):
        pass

    @abstractmethod
    def visit_section(self, section: Section):
        pass


class NaiveSemanticVisitor(Visitor):
    def visit_caption(self, caption):
        return caption.content()

    def visit_footnote(self, footnote):
        return footnote.content()

    def visit_formula(self, formula):
        return formula.content()

    def visit_list_item(self, list_item):
        return list_item.content()

    def visit_page_footer(self, page_footer):
        return page_footer.content()

    def visit_page_header(self, page_header):
        return page_header.content()

    def visit_picture(self, picture):
        raise Exception("Picture semantic output not implemented")

    def visit_section_header(self, section_header):
        return section_header.content()

    def visit_table(self, table: Table):
        content = table.content()
        return content.to_csv()

    def visit_text(self, text: Text):
        contents = text.content()
        cur = text.continued()
        while cur:
            contents = contents.rstrip() + " " + cur.content()
            cur = cur.continued()

        return contents

    def visit_title(self, title: Title):
        return title.content()

    def visit_group(self, group: Group):
        # merge content in the group naive
        contents = [child.accept(self) for child in group.children()]
        return " ".join(contents)

    def visit_section(self, section: Section):
        # merge content in the section naive
        contents = [child.accept(self) for child in section.children()]
        return " ".join(contents)


class Page:
    """
    Holds page objects not in the structure tree like page header, page footer
    and footnote.
    """

    def __init__(self, nodes: list[Node]):
        self._nodes = nodes


class Tokenizer:
    @abstractmethod
    def limit(self) -> int:
        pass

    @abstractmethod
    def count(self, semantic) -> int:
        pass


class Document:
    def __init__(self, root, nodes, pages, properties):
        self._root = root
        self._nodes = nodes
        self._pages = pages
        self._properties = properties

    def summary(self):
        pass

    def filter(self):
        pass

    def chunk(self, visitor, tokenizer):
        """
        First pass use tokenizer to calculate token count for each node, second
        round do traverse the tree and collect the node with token size smaller
        than limit
        @param visitor: a visitor defines how text is assembled from the tree
        @param tokenizer: a tokenizer defines max token limit
        """
        token_dict = {}
        chunks = []

        def post_traverse(node):
            if isinstance(node, Internal):
                for child in node.children():
                    post_traverse(child)
                semantic = node.accept(visitor)
                count = tokenizer.count(semantic)
                token_dict[node.node_id()] = (count, semantic)
            else:
                semantic = node.accept(visitor)
                count = tokenizer.count(semantic)
                token_dict[node.node_id()] = (count, semantic)

        def pre_traverse(node):
            count, semantic = token_dict[node.node_id()]
            if count <= tokenizer.limit():
                chunks.append(semantic)
            elif isinstance(node, Internal):
                for child in node.children():
                    pre_traverse(child)
            else:
                chunks.append(semantic)

        post_traverse(self._root)
        pre_traverse(self._root)

        return chunks
