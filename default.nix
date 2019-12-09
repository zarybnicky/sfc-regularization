let
  jupyter = import (builtins.fetchGit {
    url = https://github.com/tweag/jupyterWith;
    rev = "119c85563631998281a3eb8e596128e06e66752d";
  }) {};

  ihaskell = jupyter.kernels.iHaskellWith {
    name = "haskell";
    packages = p: with p; [
      hvega
      formatting
    ];
  };

  ipython = jupyter.kernels.iPythonWith {
    name = "python";
    packages = p: with p; [
      numpy
      scipy
      pandas
      matplotlib
      seaborn
      umap-learn
      scikitlearn
      tabulate
    ];
  };

  jupyterEnvironment = jupyter.jupyterlabWith {
    extraPackages= p: with p.python37Packages; [
      jupyter_console
      matplotlib
      pandas
      tabulate
    ];
    kernels = [ ihaskell ipython ];
  };
in jupyterEnvironment.env
