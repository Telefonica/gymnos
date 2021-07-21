#
#
#   Auto yaml
#
#

import re
import inspect

from typing import List
from docutils import nodes
from benedict import benedict
from omegaconf import OmegaConf
from docutils import statemachine
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
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


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


class AutoYAMLDirective(SphinxDirective):

    required_arguments = 1
    has_content = False
    option_spec = {
        'key': directives.unchanged,
        "caption": directives.unchanged
    }

    def run(self):
        yaml_path = self.arguments[0]

        config = OmegaConf.load(yaml_path)

        rst_content = inspect.cleandoc("""
        .. code-block:: yaml
        """)

        if "caption" in self.options:
            caption = self.options["caption"]

            pattern = r"(.*){(.+)}(.*)"

            match = re.match(pattern, caption)
            if match:
                key_ref = match.group(2)
                replace_val = benedict(OmegaConf.to_object(config))[key_ref]
                caption = re.sub(pattern, "\\1" + replace_val + "\\3", caption)

            rst_content += f"\n    :caption: {caption}"

        if "key" in self.options:
            config = config[self.options['key']]

        content = OmegaConf.to_yaml(config)

        content = content.replace('\n', '\n' + " " * 4)

        rst_content = rst_content + "\n\n" + (" " * 4) + content

        node = convert_rst_to_nodes(self, rst_content)

        return node


class AutoYamlDocstring(SphinxDirective):

    required_arguments = 1
    has_content = False
    option_spec = {
        'lineno-start': directives.nonnegative_int
    }

    def run(self):
        yaml_path = self.arguments[0]

        lineno_start = self.options.get("lineno-start", 0)

        docstring = ""

        with open(yaml_path, "r") as fp:
            for _ in range(lineno_start):
                next(fp)

            for line in fp:
                if line.startswith("#"):
                    line = remove_prefix(line, "#")
                    line = remove_prefix(line, " ")
                    docstring += line
                else:
                    break

        node = convert_rst_to_nodes(self, docstring)

        return node


def setup(app):
    app.add_directive('autoyaml', AutoYAMLDirective)
    app.add_directive("autoyamldoc", AutoYamlDocstring)
