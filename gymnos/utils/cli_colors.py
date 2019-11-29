#
#
#   CLI Colors
#
#

# Adapted from https://github.com/ines/wasabi

COLORS = {
    "good": 2,
    "fail": 1,
    "warn": 3,
    "info": 4,
    "red": 1,
    "green": 2,
    "yellow": 3,
    "blue": 4,
    "pink": 5,
    "cyan": 6,
    "white": 7,
    "grey": 8,
    "black": 16,
}


def color(text, fg=None, bg=None, bold=False, underline=False):
    """
    Color text by applying ANSI escape sequence.
    Parameters
    ------------
    text: str
        The text to be formatted
    fg: str or int
        Foreground color. String name or 0 - 256 (see COLORS).
    bg: str or int
        Background color. String name or 0 - 256 (see COLORS).
    bold: bool
        Whether or not format text in bold
    underline: bool
        Whether or not underline text

    Returns
    -------
    str
        Formatted text
    """
    fg = COLORS.get(fg, fg)
    bg = COLORS.get(bg, bg)
    if not any([fg, bg, bold]):
        return text
    styles = []
    if bold:
        styles.append("1")
    if underline:
        styles.append("4")
    if fg:
        styles.append("38;5;{}".format(fg))
    if bg:
        styles.append("48;5;{}".format(bg))
    return "\x1b[{}m{}\x1b[0m".format(";".join(styles), text)
