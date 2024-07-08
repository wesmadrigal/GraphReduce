#!/usr/bin/env python

import typing

import typer

from .auto_fe import auto_fe_cli


entrypoint_cli_typer = typer.Typer(
        no_args_is_help=True,
        add_completion=False,
        rich_markup_mode="markdown",
        help="""
        See examples at https://github.com/wesmadrigal/graphreduce
        """
)

# Automated feature engineering
entrypoint_cli_typer.add_typer(auto_fe_cli, rich_help_panel="autofe")
entrypoint_cli = typer.main.get_command(entrypoint_cli_typer)
entrypoint_cli.list_commands(None)


if __name__ == '__main__':
    entrypoint_cli()
