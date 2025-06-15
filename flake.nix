{
  description = "NanoDQN - DQN Scaling Laws BSc Thesis";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        python = pkgs.python311;
        
        pythonEnv = python.withPackages (ps: with ps; [
          # Core dependencies from pyproject.toml
          wandb
          jax
          jaxlib
          optax
          tqdm
          equinox
          jaxtyping
          pillow
          seaborn
          
          # Additional useful packages for development
          pip
          setuptools
          wheel
          
          # For Jupyter notebooks if needed
          jupyter
          ipython
          
          # Testing and linting
          pytest
          ruff
        ]);
        
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            uv
            
            # CUDA support (if available)
            cudatoolkit
            cudnn
            
            # Development tools
            git
            
            # For building some Python packages
            pkg-config
            gcc
            
            # Additional libraries that might be needed
            zlib
            libGL
            glib
          ];
          
          shellHook = ''
            echo "NanoDQN development environment"
            echo "Python: $(python --version)"
            echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"
            
            # Set up environment variables for CUDA if available
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.cudnn}/lib:$LD_LIBRARY_PATH
            
            # Install the package in development mode
            if [ -f "pyproject.toml" ]; then
              echo "Installing package in development mode..."
              pip install -e .
            fi
          '';
          
          # Environment variables
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.zlib
            pkgs.cudatoolkit
            pkgs.cudnn
          ];
        };
        
        # Package definition
        packages.default = python.pkgs.buildPythonPackage rec {
          pname = "nanodqn";
          version = "0.3.0";
          
          src = ./.;
          
          pyproject = true;
          
          build-system = with python.pkgs; [
            setuptools
            wheel
          ];
          
          dependencies = with python.pkgs; [
            wandb
            jax
            jaxlib
            optax
            tqdm
            equinox
            jaxtyping
            pillow
            seaborn
          ];
          
          # Skip tests for now since we don't know the test setup
          doCheck = false;
          
          meta = with pkgs.lib; {
            description = "An implementation for DQN Scaling Laws BSc Thesis";
            license = licenses.mit;
            maintainers = [ ];
          };
        };
      });
}