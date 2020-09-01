import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robusta-Eitan.Hemed", # Replace with your own username
    version="0.0.1",
    author="Eitan Hemed",
    author_email="Eitan.Hemed@gmail.com",
    description="Statistical analysis package in Python",
    long_description="""robusta is a statistical analysis package in Python,
     based on R""",
    long_description_content_type="text/markdown",
    url="https://github.com/EitanHemed/robusta",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)