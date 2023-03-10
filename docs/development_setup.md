# Development setup

This is a practical guide for developers wanting to develop within cogment verse.

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

Cogment verse includes a web app designed for human-in-the-loop learning developed with React.

To develop the web app, you'll need to install [Node.JS v16](https://nodejs.org/) or above.

Sources for the web app can be found in `/cogment_verse/web/web_app`.

### Prebuilt web app - Default

When running a default instance of cogment verse, the prebuilt web app, located in `/cogment_verse/web/web_app/build` is used. e.g.

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

### Development "autoreload" mode

To start an autoreloading isntance of the webapp, set `services.web.dev` to `True`. e.g.

```console
$ python -m main service.web.dev=True
```

The web app will be served as an autoreloading server: any edit to the web app sources will be taken into account and cause a reload.

## Testing

### Python test suite

Run the test suite on the python codebase using `pytest` by running the following in the virtual environment in the root directory:

```console
$ python -m pytest
```
