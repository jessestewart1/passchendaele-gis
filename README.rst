*****************
Passchendaele GIS
*****************

Description
===========

Code resources used for the dissertation "Contextualizing the Geographic Influence on Infantry Manoeuvrability in a
Historical Battlefield Using GIS: A Case Study of the Canadian Corps in the Second Battle of Passchendaele, First World
War" by Jesse Stewart (2025). Part of the Master of Science in Geographical Information Science program at Lund
University.

Refer to the LICENSE.rst file for disclaimer and usage restrictions.

Software Dependencies
=====================

`Git <https://git-scm.com/downloads>`_: Version control system for repository management. Allows you to clone a remote
repository (GitHub) and integrate changes into your local copy.

`conda <https://github.com/conda-forge/miniforge>`_: Package and virtual environment management system. Miniforge3 is
the preferred installer for ``conda`` due to being a minimal installer and specific to the ``conda-forge`` channel.

``conda`` installation may require the following additions to the ``Path`` environment variable in order for ``conda``
to be recognized as a valid command: `C:\\ProgramData\\miniforge3\condabin`. The exact location of this directory may
vary depending on your setup.

Setup
=====

1. Clone repository::

    git clone https://github.com/jessestewart1/passchendaele-gis.git

2. Install ``conda`` virtual environment (path depends on git clone destination)::

    conda env create -f C:/passchendaele-gis/environment.yml

Usage
=====

All scripts are intended to be executed within a ``conda`` virtual environment, which must be activated before executing
any scripts in order to make use of the contained dependencies::

    conda activate passchendaele-gis

All scripts are implemented as CLIs (command-line interfaces), which can be called from any terminal. The usage
instructions and inputs to each script can be viewed by passing ``--help`` to any script::

    python script_name --help
