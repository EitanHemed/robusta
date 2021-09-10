import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="robusta_stats",
    version="0.0.2",
    author="Eitan Hemed",
    author_email="Eitan.Hemed@gmail.com",
    description="Statistical analysis package in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EitanHemed/robusta",
    download_url="https://github.com/EitanHemed/robusta",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.10',
    packages=setuptools.find_packages()
)
