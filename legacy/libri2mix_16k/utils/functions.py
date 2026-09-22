"""Compatibility import omitted from the packaged course-project source.

The old engine imports utils.functions, but the supplied ZIP places the actual
implementation in utils/implements/functions.py. Re-export that exact function.
"""
from .implements.functions import apply_cmvn
