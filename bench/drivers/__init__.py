"""Subprocess drivers: one per tool that is a Python library rather than a CLI.

Each driver stitches/registers one region and writes the shared ``positions.json``
contract into its output directory, so the runner spawns and measures one uniform
kind of thing regardless of how different the tools are underneath.
"""
