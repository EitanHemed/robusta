import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
with open('./requirements.txt') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="robusta-Eitan.Hemed", # Replace with your own username
    version="0.0.1",
    author="Eitan Hemed",
    author_email="Eitan.Hemed@gmail.com",
    description="Statistical analysis package in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EitanHemed/robusta",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)