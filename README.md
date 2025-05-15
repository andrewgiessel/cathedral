# Cathedral

<p align="center">
  <img src="assets/logo.jpeg" alt="Cathedral Logo" width="300"/>
</p>

[![Release](https://img.shields.io/github/v/release/andrewgiessel/cathedral)](https://img.shields.io/github/v/release/andrewgiessel/cathedral)
[![Build status](https://img.shields.io/github/actions/workflow/status/andrewgiessel/cathedral/main.yml?branch=main)](https://github.com/andrewgiessel/cathedral/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/andrewgiessel/cathedral)](https://img.shields.io/github/commit-activity/m/andrewgiessel/cathedral)

## About

Cathedral is an implementation of the Church programming language in Hy. Church is a probabilistic programming language based on lambda calculus and designed for describing generative processes and reasoning about generated observations.

By implementing Church in Hy (a Lisp dialect embedded in Python), Cathedral provides the expressiveness of Church's probabilistic programming model with access to Python's rich ecosystem of scientific and machine learning libraries.

- **Github repository**: <https://github.com/andrewgiessel/cathedral/>
- **Documentation** <https://andrewgiessel.github.io/cathedral/>

## Features

- Stochastic memoization
- Inference algorithms like Metropolis-Hastings, Gibbs sampling, and particle filtering
- Expressive and concise syntax
- Seamless integration with Python scientific libraries

## References

- [Church: A Language for Generative Models](https://arxiv.org/pdf/1206.3255v2.pdf) - Goodman et al.
- [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- [Hy: A Lisp dialect embedded in Python](https://github.com/hylang/hy)
- [Probabilistic Models of Cognition](http://probmods.org/) - Goodman & Tenenbaum
- [Rational Implementations of Probabilistic Programs](https://homes.luddy.indiana.edu/ccshan/rational/dsl-paper.pdf) - Shan & Ramsey
- [Gamble: A library for probabilistic programming](https://rmculpepper.github.io/gamble/prob-bibliography.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Installation

```bash
pip install cathedral
```

## Quick Example

```python
# Example code coming soon
```

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:andrewgiessel/cathedral.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

---

Repository initiated with [profitablesignals/cookiecutter-uv](https://github.com/profitablesignals/cookiecutter).
