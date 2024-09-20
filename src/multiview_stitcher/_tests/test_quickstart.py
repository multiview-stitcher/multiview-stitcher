# Test the quickstart code in the README


import os
from pathlib import Path

from multiview_stitcher import _tests

tests_path = Path(_tests.__file__).parent
quickstart_code_dir = tests_path / "quickstart"


def test_quickstart_code():
    """
    Test the quickstart code
    """

    for block_filepath in sorted(os.listdir(quickstart_code_dir)):
        if block_filepath.endswith(".py"):
            with open(quickstart_code_dir / block_filepath) as f:
                block_code = f.read()
            exec(block_code, globals())

    assert len(fused_sim.data.shape)  # noqa: F821


def test_quickstart_code_sync():
    """
    Make sure the code in src/multiview_stitcher/_tests/quickstart
    is in

    - the README
    - docs/code_example.md

    and didn't get out of sync.
    """

    md_paths = [
        tests_path / "../../.." / "docs" / "code_example.md",
        tests_path / "../../.." / "README.md",
    ]
    md_strings = []
    for md_path in md_paths:
        with open(md_path) as f:
            md_strings.append(f.read())

    for block_filepath in sorted(os.listdir(quickstart_code_dir)):
        if block_filepath.endswith(".py"):
            with open(quickstart_code_dir / block_filepath) as f:
                block_code = f.read()
            for md_string in md_strings:
                assert block_code in md_string
