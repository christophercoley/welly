# Installation

At the command line:

    $ pip install welly

Or, if you use Conda environments:

    $ conda create -n welly python=3.10
    $ conda activate welly
    $ pip install welly

## Optional Dependencies

### DLIS Support

To read DLIS files (Digital Log Interchange Standard), install with the `dlis` extra:

    $ pip install welly[dlis]

This installs the `dlisio` library which is required for `Well.from_dlis()`.

## Development Installation

If you want to help develop Welly, read [Development](development.md).
