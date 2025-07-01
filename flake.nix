{
  description = "A Unified Scaling Law of Bootstrapped DQNs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python312;

        pythonEnv = python;

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python
            uv

            # CUDA support (if available)
            cudatoolkit

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

            # Set up environment variables for CUDA if available
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.stdenv.cc.cc.lib}/lib/:$LD_LIBRARY_PATH

            # Install the package in development mode using uv
            if [ -f "pyproject.toml" ]; then
              echo "Installing/syncing Python dependencies with uv..."
              uv sync
              echo "Dependencies installed."
              echo "JAX version: $( uv run python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'jax not found')"
            fi
          '';

          # Environment variables
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.zlib
            pkgs.cudatoolkit
          ];
        };

        # Package definition
        packages.default = python.pkgs.buildPythonPackage rec {
          pname = "unified-bootstrapped-dqn-scaling";
          version = "1.0.0";

          src = ./.;

          pyproject = true;

          # build-system and dependencies are managed by uv, see devShells.default.shellHook

          # Skip tests for now since we don't know the test setup
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Official implementation of 'A Unified Scaling Law for Bootstrapped DQNs'";
            license = licenses.mit;
            maintainers = [
              {
                name = "Roman Knyazhitskiy";
                email = "dqn.scaling.laws@knyaz.tech";
              }
            ];
          };
        };
      });
}
