# Development setup

This is a practical guide for developers wanting to develop within Cogment Verse.

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

## PDB debugger with multiprocessing

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
   `python ForkedPdb().set_trace() `

NOTE: The other commands are the same as the [PDB debugger](https://docs.python.org/3/library/pdb.html).

## Developing the web app

Cogment Verse includes a modular web app designed for human-in-the-loop learning developed with React.

To develop the web app, you'll need to install [Node.JS v16](https://nodejs.org/) or above.

The _host_ web app is included in the "core" cogment_verse package in `/cogment_verse/web/web_app`.

Each environment provides its own web app _plugin_ that defines how it is rendered and how the player can interact with it. Environment _plugins_ are React components that leverages the `@cogment/cogment_verse` SDK defined in `/cogment_verse/web/web_app/src/shared`. Full documentation on how to develop your own environment plugin is a work in progress, in the meantime take a look at the built-in plugins e.g. in `/environments/gym/web`.

### Prebuilt web app - Default

When running a default instance of Cogment Verse, the prebuilt web app, located in `/cogment_verse/web/web_app/dist` is used. e.g.

```console
$ python -m main
```

> ⚠️ in this mode, any changes to the sources will be ignored

### Trigger a rebuild

To trigger a rebuild before launching the webapp, the `services.web.build` needs to be set to `True`. e.g.

```console
$ python -m main services.web.build=True
```

This perform to a full static build of the web app before launching the instance.

> ⚠️ only the _host_ webapp is rebuilt in that way, changes to the environment _plugins_ need to be rebuilt separately, usually by running `npm run build` in their directory.

## Testing

### Python test suite

Run the test suite on the python codebase using `pytest` by running the following in the virtual environment in the root directory:

```console
$ python -m pytest
```

### Functional Tests

To only run functional tests

```console
$ python -m pytest -m functional --durations=0 --no-header -v
```
