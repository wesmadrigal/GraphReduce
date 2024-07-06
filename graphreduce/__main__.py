#!/usr/bin/env python

import sys

import typer

from .cli.entry_point import entrypoint_cli




def main():

    if sys.version_info[:3] == (3, 8):
        pass


    try:
        entrypoint_cli()
    except Exception as exc:
        tb = exc.__cause__.__traceback__
        print(tb)


if __name__ == '__main__':
    main()
