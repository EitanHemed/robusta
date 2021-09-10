.. _installation:

Installation
============

How to install robusta?
***********************
Currently robusta is expected to work only on linux (not tested on MacOSX), given
the difficulties in installing 3.x versions of rpy2 on Windows (see `here <https://github.com/rpy2/rpy2>`_).

**How to install**

1. Make sure `R <https://www.r-project.org/>`_ R >=3.5 is installed (e.g. run
:code:`$ R -e "version" | grep "version.string"` in your shell).

2. :code:`$ pip install robusta-stats`. Since on Linux R libraries are usually built from
source, installation could take a while on the first time you run it (unless your R installation
already includes some of the dependencies).