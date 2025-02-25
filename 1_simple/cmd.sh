docker build -t llm-test .
docker run --rm llm-test

# CUDA

docker run --gpus all --rm llm-test