FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image currently ships a Yarn apt source with a missing signing key.
# Remove it so feature installers that run apt-get update (e.g., docker-in-docker) succeed.
RUN rm -f /etc/apt/sources.list.d/yarn.list
