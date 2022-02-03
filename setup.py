from setuptools import setup, find_packages
import os
import re


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1])
                )
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


# requirements = resolve_requirements(os.path.join(os.path.dirname(__file__),
#                                                  'requirements.txt'))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))
license = read_file(os.path.join(os.path.dirname(__file__), "LICENSE"))

setup(
    name="medhack",
    version="0.0.1",
    packages=find_packages(),
    # url='path/to/url',
    test_suite="pytest",
    long_description=readme,
    long_description_content_type="text/markdown",
    # install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.8",
    # author="FridgeReloaded",
    # author_email="fridge.reloaded@dkfz.de",
    license=license,
)
