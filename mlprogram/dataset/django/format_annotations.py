import re
from typing import List


def format_annotations(annotations: List[str]) -> List[str]:
    """
    Preprocess the list of annotation in Django dataset.

    Parameters
    ----------
    annotations: List[str]
        The list of lines in Django dataset

    Returns
    -------
    List[str]
        The formatted lines of the annotations
    """
    annots: List[str] = []
    for annotation in annotations:
        m = re.search(r"^.*\.   ", annotation)
        if m is None or len(annots) == 0:
            annots.append(annotation.strip())
        else:
            annots[-1] = f"{annots[-1]} {m.group(0)}".strip()
            annots.append(annotation[m.end():].strip())
    return list(filter(lambda x: len(x) > 0, annots))
