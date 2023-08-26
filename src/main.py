import click

from commands import get_cli


def main():
    cli = get_cli()
    click.echo('--- CLI started ---')
    cli()


if __name__ == '__main__':
    main()
