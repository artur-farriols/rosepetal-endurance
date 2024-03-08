#########################################################
## Python Environment with CUDA
#########################################################

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

FROM ultralytics/ultralytics:latest

LABEL maintainer="Rosepetal"

# Setup proxies
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV DEBIAN_FRONTEND="noninteractive"

# Update system and install necessary packages
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        # python3.8 \
        # python3-pip \
        # python3-dev \
        # curl \
        # wget \
        # npm \
        # git \
        # software-properties-common \
        libgomp1 \
        libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Create an empty directory called 'models' inside /app/test_docker/
# RUN mkdir -p /app/test_docker/models

# Copy required files and directories
COPY rosepetal-grpc/grifols_server /app/grifols_server
COPY paddleclas-0.0.0-py3-none-any.whl \
     requirements.txt \
     rosepetal-0.0.1-py3-none-any.whl \
     paddle_test.py \
     /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir paddleclas-0.0.0-py3-none-any.whl \
    && pip3 install --no-cache-dir rosepetal-0.0.1-py3-none-any.whl \
    && rm paddleclas-0.0.0-py3-none-any.whl rosepetal-0.0.1-py3-none-any.whl grifols_server/Install.md grifols_server/requirements.txt \
    && rm -rf grifols_server/demo/

# Execute server_test.py with -u option
CMD ["python3", "-u", "/app/grifols_server/server.py"]
# CMD ["python3", "-u", "/app/paddle_test.py"]
