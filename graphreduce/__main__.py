#!/usr/bin/env python

import sys

import typer

from .cli.entry_point import entrypoint_cli


def main():
    if sys.version_info[:3] == (3, 8):
        pass
    entrypoint_cli()


if __name__ == '__main__':
    main()
