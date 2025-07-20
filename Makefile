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
	npm run serve

# Clean up the project
clean:
	rm -rf dist $(NODE_MODULES)
