FROM nvcr.io/nvidia/jax:25.04-py3

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Install the package and dependencies
RUN uv sync

# Set the default command to drop into the workspace directory
CMD ["/bin/bash"]
