Generally useful adjustments and additions to PyIBL are welcome. Please clone
the repository to your own machine, make a suitable branch, and when finished make a pull request (that’s
the usual term in GitHub, but some  GitHub alternatives use the far clearer term “merge request” which
is what you’re really doing!) on GitHub in the usual way,
to ensure review and, if necessary, discussion of your changes.
Please do ***not*** make *any* changes to the `main` branch of this repository—always make a pull request instead;
in addition, unless there is a compelling reason to do so, please do not push any branches to the
repository, except for those ready to be merged.
Before making a pull request, please ensure

- that *all* the unit tests pass

- that suitable unit tests have been added for any additions you make, or to test for regressions of any bugs
  you have fixed

- and that the documentation has been suitably revised, if necessary, and *builds correctly*.

In addition, PyACTUp is at the heart of PyIBL. Because of this many possible modifications to PyIBL may actually
need to be modifications to PyACTUp. See the file `contributing.md` in the PyACTUp repository for advice on
modifying it, if necessary.

Depending upon the changes you are making, some further things to consider:

- Should your changes also be reflected in `details` and `trace`?

- Do any API changes closely match the current API choices?

- Do all changes work correctly with both options with and without attributtes?

- Do all undocumented or otherwise non-public functions, methods or other members’ names start with an underscore?

- When adding a new function, method or similar, not only do you need to include a suitable docstring in the definition
  itself, it needs to be cited in `doc/index.rst`

- If you need to import any modules not already imported by PyIBL, please be sure to update `setup.py`
  and `requirements.txt` to match.

- Ensure appropriate errors, worded in ways helpful to the end user, are raised, and document all errors
  that can be raised in the docstring.

- Do your changes need to be carefully tested with `optimized_learning`, delayed feedback and`fixed_noise`?
