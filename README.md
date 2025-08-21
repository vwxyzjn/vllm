# Extract Pytorch Nightly Wheels from vLLM Docker Container

This guide shows how to extract wheel files from the vLLM nightly Docker container into a local directory.

## Commands to Extract Wheels

### 0. Build the vLLM Docker Image

```bash
sudo docker build -t vllm-nightly-2 -f docker/Dockerfile.nightly_torch .
```


### 1. Create local wheels directory
```bash
mkdir -p ./wheels
```

### 2. Run container with mounted wheels directory and copy wheels
```bash
# Run the container with the local ./wheels directory mounted
docker run --rm -v "$(pwd)/wheels:/host-wheels" vllm-nightly-2 bash -c "
    echo 'Copying vLLM wheels...' && \
    cp -v /vllm-workspace/vllm-dist/*.whl /host-wheels/ 2>/dev/null || echo 'No vLLM wheels found' && \
    echo 'Copying XFormers wheels...' && \
    cp -v /vllm-workspace/xformers-dist/*.whl /host-wheels/ 2>/dev/null || echo 'No XFormers wheels found' && \
    echo 'Copying FlashInfer wheels...' && \
    cp -v /vllm-workspace/flashinfer-dist/*.whl /host-wheels/ 2>/dev/null || echo 'No FlashInfer wheels found' && \
    echo 'Wheel extraction completed!' && \
    cp -v /vllm-workspace/torch_build_versions.txt /host-wheels/ 2>/dev/null || echo 'No torch_build_versions.txt found' && \
    echo 'torch_build_versions.txt copied to host!' && \
    echo 'Available wheels in container:' && \
    find /vllm-workspace -name '*.whl' -type f 2>/dev/null || echo 'No wheels found' && \
    echo 'Wheels copied to host:' && \
    ls -la /host-wheels/*.whl 2>/dev/null || echo 'No wheels copied'
"
```

### 3. Verify extracted wheels
```bash
# List the extracted wheels
ls -la ./wheels/
```

## Alternative: Interactive Method

If you prefer to run the container interactively:

```bash
# Start an interactive session with mounted wheels directory
docker run --rm -it -v "$(pwd)/wheels:/host-wheels" vllm-nightly-2 bash

# Inside the container, copy the wheels manually:
cp /vllm-workspace/vllm-dist/*.whl /host-wheels/
cp /vllm-workspace/xformers-dist/*.whl /host-wheels/
cp /vllm-workspace/flashinfer-dist/*.whl /host-wheels/

# Exit the container
exit
```

## What Wheels Are Available?

The container typically contains:

- **vLLM wheels** (`/vllm-workspace/vllm-dist/`): Core vLLM package wheels
- **XFormers wheels** (`/vllm-workspace/xformers-dist/`): Memory-efficient attention wheels  
- **FlashInfer wheels** (`/vllm-workspace/flashinfer-dist/`): High-performance inference wheels

## Usage Notes

- The `./wheels` directory will be created if it doesn't exist
- The mount point `/host-wheels` inside the container maps to your local `./wheels` directory
- Wheels are copied with verbose output (`-v`) to show what's being copied
- Error handling ensures the command completes even if some wheel directories are missing
- The container is automatically removed (`--rm`) after execution
