#
#
#   I/O Utils
#
#


def read_file_text(file_path):
    """
    Read file text.

    Parameters
    ----------
    file_path: str
        File path to read.

    Returns
    -------
    file_text: str
        Text.
    """
    with open(file_path) as f:
        return f.read()


def read_lines(file_path):
    """
    Read file lines

    Parameters
    -----------
    file_path: str
        File path to read lines.

    Returns
    -------
    lines: list of str
        Text for each line.
    """
    with open(file_path) as archive:
        names = [line.rstrip('\n') for line in archive]

    return names
