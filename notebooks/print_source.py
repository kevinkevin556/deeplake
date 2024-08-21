import inspect
import re

from IPython.display import HTML, display
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import Python3Lexer


def remove_method(source_code: str, method_name: str) -> str:
    pattern = rf"    def {method_name}\(.*\):\n(?:        .*\n)*"
    modified_code = re.sub(pattern, "", source_code)
    return modified_code


def print_source(cclass, omit=[]):
    source_code = inspect.getsource(cclass)
    for m in omit:
        source_code = remove_method(source_code, m)

    formatter = HtmlFormatter(nobackground=False, style="lightbulb")
    display(HTML(f'<style>{ formatter.get_style_defs(".highlight") }</style>'))
    display(HTML(data=highlight(source_code, Python3Lexer(), HtmlFormatter())))
