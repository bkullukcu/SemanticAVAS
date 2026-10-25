# Contributing

Bug reports should include the package version, Python version, the command that failed and a small input that reproduces the problem. Remove confidential measurement data and participant identifiers before attaching files.

For code changes, open an issue describing the numerical or measurement problem first. Add a test that exercises the change, run `python -m pytest -q`, and document any change to array dimensions, units or statistical assumptions.

The API follows the research-preview release series. Check the changelog when updating between minor versions.
