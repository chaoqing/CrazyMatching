# Makefile for CrazyMatching

# Variables
NODE_MODULES = node_modules

# Phony targets (targets that don't represent files)
.PHONY: all install dev build serve clean

all: install dev

# Install dependencies
install: $(NODE_MODULES)

$(NODE_MODULES): package-lock.json
	npm install
	touch $(NODE_MODULES)

# Start the development server
dev:
	npm run dev

# Build the project for production
build:
	npm run build

# Serve the production build
serve: build
	npm run serve --host

# Serve the production build over HTTPS using localtunnel
# The default port for `vite preview` (used by `npm run serve`) is 4173.
# If your server runs on a different port, adjust the localtunnel command accordingly.
serve-https: serve localtunnel
localtunnel: build
	@echo "Waiting for local server to start with password $$(env https_proxy= curl -4 https://ipecho.net/plain) ..."
	@sleep 1 # Give the server a few seconds to start
	@echo "Starting localtunnel tunnel..."
	@env https_proxy= $(NODE_MODULES)/.bin/lt --port 4173

# Clean up the project
clean:
	rm -rf dist $(NODE_MODULES)
