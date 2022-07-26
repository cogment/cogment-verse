# Development setup

This is a practical guide for developers wanting to develop within cogment verse.

## Additional dependencies

To support development, additional dependencies are required.

- [Node.JS v14](https://nodejs.org/) or above is required to develop the web components of cogment verse.

> 🚧 _in construction_ 🚧

## Linting

### Python code formatting

Check the code style using `black` by running the following in the virtual environment in the root directory:

```console
$ black --check --diff .
```

Fix the code style by running:

```console
$ black .
```

### Python code quality

Check the code quality using `pylint` by running the following in the virtual environment in the root directory:

```console
$ pylint --recursive=y .
```

## PDB debugger with multiptrocessing 
1. Create the following python script `md_debug.py`
    ```python
    import sys
    import pdb

    class ForkedPdb(pdb.Pdb):
        """A Pdb subclass that may be used from a forked multiprocessing child"""

        def interaction(self, *args, **kwargs):
            _stdin = sys.stdin
            try:
                sys.stdin = open('/dev/stdin')
                pdb.Pdb.interaction(self, *args, **kwargs)
            finally:
                sys.stdin = _stdin
    ```
2. Import the above script to the python file that you want to debug
    ```python
    from mp_debug import ForkedPdb 
    ```
3. Set a breakpoint using the following command
    ```python
    ForkedPdb().set_trace()
    ```
NOTE: The other commands are the same as the [PDB debugger](https://docs.python.org/3/library/pdb.html).


## Testing

### Python test suite

Run the test suite on the python codebase using `pytest` by running the following in the virtual environment in the root directory:

```console
$ python -m pytest
```
