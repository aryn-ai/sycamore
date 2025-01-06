# Contributing to Sycamore

We welcome contributions to Sycamore.

## How to Contribute

To avoid duplicate or unnecessary work, please follow this process to make contributions:

- Find or create an issue in GitHub. Leave a note on an existing issue to let the community know you are working on it, or create a new issue for a feature you would like to add. If there are significant design decisions to cover, please start a conversation in the issue to get feedback early.
- Create a fork of Sycamore and start development in a feature branch. You can find information about how to set up Sycamore for development below. More information about development patterns in Github can be found [here](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).
- Make sure linting rules and all existing unit tests pass. Consider whether your changes need new tests, and make sure those pass as well.
- Larger features may need new documentation. When in doubt, start a discussion on the associated issue.
- Create a pull request against the `main` branch of this repository. Pull requests are lightweight, and we encourage you to create a draft PR early to get feedback.

## What to Contribute

Sycamore is a new project with big ambitions, so there are a lot of ways to contribute:

- **Bug reports**. We love bug reports! If you find something broken, please open an issue using the Bug report template and provide as much detail as possible about what happened and how we can reproduce it.
- **Documentation improvements**. We want it to be as easy as possible to get started with sycamore. If you spot something wrong or something that could be clearer in the docs, please open an issue, or better yet contribute a fix yourself!
- **New sources and file formats**. Sycamore can be used to process all types of unstructured data. If there is a format or source that we're missing, consider adding it and contributing back.
- **New models and LLMs**. Similarly, we want to support a wide variety of machine learning models and LLMs for different purposes, such as segmentation, embedding, and entity extraction.
- **New transformations**. Transformations are the core of Sycamore. If you have an idea for a useful transformation, create an issue and get started!

These are just a few ideas. For more, check out our list of [issues](https://github.com/aryn-ai/sycamore/issues).

## Developer Setup

Sycamore targets Python 3.9+ and runs primarily on Mac and Linux. The following will help you set up your environment.

### Dependency Management

We use poetry to manage Python dependencies in Sycamore. You can install poetry using the instructions [here](https://python-poetry.org/docs/#installing-with-the-official-installer). Once you have poetry installed, you can install all dependencies by running

```bash
poetry install --all-extras --no-root

```

In addition, some pdf processing methods require [Poppler](https://poppler.freedesktop.org/). You can install this with the OS-native package manager of your choice. For example, the command for Homebrew on Mac OS is

```bash
brew install poppler
```

### Linting

We use `ruff` to lint sycamore and `black` to automatically format our code. You can run these tools from the root of the repository using

```bash
poetry run ruff check .
poetry run black .

```

Note that `black` will reformat code in place. Make sure to include these changes in your commits.

### Type Checking

We use type annotations in Sycamore and use `mypy` to perform static type-checking. You can run this using

```bash
poetry run mypy --install-types .

```

Currently `mypy` will fail with the `--strict` flag, but it should pass cleanly without.

### Testing

Our tests are implemented using PyTest. You can run unit and integration tests as follows:

```bash
# Unit Tests
poetry run pytest lib/sycamore/sycamore/tests/unit/

# Integration Tests

# Warning: as of 2024-05-17 the integration tests are currently broken. We are working on fixing them.
# poetry run pytest lib/sycamore/sycamore/tests/integration/

# All Tests
poetry run pytest

```

Note that the integration tests currently require the `OPENAI_API_KEY` environment variable to be set to a valid OpenAI key.

### Pre-Commit

Sycamore includes a pre-commit configuration that will automatically run linting, formatting, and type-checking, as well the unit tests. You can perform a one-time run of these checks using

```bash
poetry run pre-commit run --all-files
```

or install it to run on git commits using

```bash
poetry run pre-commit install

```

Similar checks run in our CI environment on new pull requests and check-ins. New features must pass the mandatory checks before they will be merged.

## Releases

Release branches will be cut from the main branch following semantic versioning. The release process is currently manual, though we plan to automate it over time.
