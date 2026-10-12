# Release validation

Validated on 2026-10-12.

| Check | Result |
| --- | --- |
| Regression and local HTTP integration tests, Windows / Python 3.12 | 20 passed |
| Same tests, Linux / Python 3.11 | 20 passed |
| Bash syntax for the three requested launchers | Passed |
| Python source distribution and wheel build | Passed |
| Wheel installation and dataset lookup from another working directory | Passed |
| Clean dataset snapshot | 5,352 records; original bytes preserved |
| Removed dataset snapshot | 3,751 records; original bytes preserved |
| Dataset SHA-256 and schema checks | Passed |
| Credential patterns and private machine path scan | No matches in distributable files |
| Exact matching against credentials in the supplied launch scripts | No matches in release source or wheel |
| Manuscript figure crops | Visually inspected; plots and captions retained |

The tests use a local HTTP server and mocked local generation. No paid external API requests or full GPU inference were performed. The original model-comparison experiments were not rerun. Upstream dataset license texts were not supplied as part of the project snapshot; attribution and licensing boundaries are documented in `DATA_LICENSE.md`.

Run `python -m pytest -q` and `python tools/check_release.py` after changes. Source archives should contain only the release tree, without original Git history, local environments, model weights, or historical results.
