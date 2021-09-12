.. _installation:

Installation
============

How to install robusta?
***********************
Currently robusta is ensured to work mostly on linux. For Windows the situation is a bit more tricky
but robusta seems to work as well. robusta was Not tested on MacOSX, but is expected to work.

**How to install**

1. Make sure the proper `R <https://www.r-project.org/>`_ version is installed.

    .. list-table:: Title
       :widths: 25 25 50
       :header-rows: 1

       * - OS
         - R version
         - Test using
       * - Linux
         - >= 3.6.3
         - :code:`$ R -e "version" | grep "version.string"`
       * - Windows
         - >= 4.0.1
         - :code:`>>> R -e "version" | findstr "version.string"`


2. robusta requires that several R packages will be installed (see ``robusta.misc.pyrio.required_r_libraries``).
To avoid changes to your existing R installation, it is recommended to use different environments, e.g., using conda.
You could use the following or skip to the next step.

    .. code-block::
      :linenos:

      conda create --name rst pip -y & activate rst
      conda install -c conda-forge r-base=4 -y
      pip install robusta-stats


3. :code:`pip install robusta-stats`.


4. Enter a python session and run :code:`import robusta as rst`. On the first
time that you will import robusta many libraries and their dependencies will be installed
(more if you begin with an empty R environment). On windows this is expected to take 4-5 minutes, on Linux R libraries
are build from source, so installation takes about 20-30 minutes.

