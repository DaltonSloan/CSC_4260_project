ARG PYTHON_VARIANT=3.12-bookworm
FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VARIANT}

# The base image currently ships a Yarn apt source with a missing signing key.
# Remove it so feature installers that run apt-get update succeed consistently.
RUN rm -f /etc/apt/sources.list.d/yarn.list

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
