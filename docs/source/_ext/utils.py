#
#
#   Extensions utils
#
#

from typing import List
from docutils import nodes
from docutils import statemachine
from sphinx.util.nodes import nested_parse_with_titles


def convert_rst_to_nodes(self, rst_source: str) -> List[nodes.Node]:
    """Turn an RST string into a node that can be used in the document."""
    node = nodes.Element()
    node.document = self.state.document
    nested_parse_with_titles(
        state=self.state,
        content=statemachine.ViewList(
            statemachine.string2lines(rst_source),
        ),
        node=node,
    )
    return node.children
