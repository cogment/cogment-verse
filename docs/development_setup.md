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

## Testing

### Python test suite

Run the test suite on the python codebase using `pytest` by running the following in the virtual environment in the root directory:

```console
$ python -m pytest
```
