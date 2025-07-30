# Makefile for CrazyMatching

# Variables
NODE_MODULES = node_modules

# Phony targets (targets that don't represent files)
.PHONY: all install dev build build-in-docker pack serve clean

all: install dev

# Install dependencies
install: $(NODE_MODULES)
.PHONY: install-npm-deps
install-npm-deps:
	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
	export NVM_DIR="$$HOME/.nvm" && [ -s "$$NVM_DIR/nvm.sh" ] && \. "$$NVM_DIR/nvm.sh" && nvm install node && nvm use node

$(NODE_MODULES): package-lock.json
	npm install
	touch $(NODE_MODULES)

# Start the development server
dev: $(NODE_MODULES)
	npm run dev

# Build the project for production
build:
	npm run build

# Serve the production build
serve: build
	npm run serve --host

dev-https: dev localtunnel
localtunnel:
	@echo "Waiting for local server to start with password $$(env https_proxy= curl -s -4 https://ipecho.net/plain) ..."
	@sleep 1 # Give the server a few seconds to start
	@echo "Starting localtunnel tunnel..."
	@env https_proxy= $(NODE_MODULES)/.bin/lt --port 5173

# Clean up the project
clean:
	rm -rf dist $(NODE_MODULES)

# Build frontend in Docker and zip the output
build-in-docker:
	@echo "Building frontend in Docker..."
	docker build -t crazy-matching-frontend .
	@echo "Copying dist from Docker container and zipping..."
	@rm -rf dist
	docker create --name temp-crazy-matching-container crazy-matching-frontend
	docker cp temp-crazy-matching-container:/usr/share/nginx/html ./dist
	docker rm temp-crazy-matching-container

pack: build-in-docker
	tar -czvf dist.tar.gz -C dist ./

deploy: pack
	@echo "Deploying to GitHub Pages..."
	test -d _deploy_temp/.git || git clone --single-branch --branch gh-pages $$(git config --get remote.origin.url) _deploy_temp
	rm -rf _deploy_temp/*
	tar -xzvf dist.tar.gz -C _deploy_temp
	cd _deploy_temp && \
	git add . && \
	git commit -m "Deploy to GitHub Pages $$(date)"
	@echo "env -C _deploy_temp git push origin gh-pages"


	

# --- Custom Model Training and Conversion --- #

.PHONY: install-python-deps train convert model-clean

# Define paths for model training and conversion
MODEL_DIR := model
TRAIN_SCRIPT := $(MODEL_DIR)/train.py
PYTHON_REQUIREMENTS := $(MODEL_DIR)/requirements.txt

SAVED_MODEL_PATH := $(MODEL_DIR)/exported_model
TFJS_OUTPUT_DIR := public/models/crazy_matching

# TensorFlow.js converter output node names for your model
# This should match the name of the output layer in your train.py (e.g., 'bbox_output')
OUTPUT_NODE_NAMES := bbox_output

# Add model-related targets to the 'all' target if you want them to run by default
# all: install dev build train convert # Uncomment if you want 'make all' to also train and convert

install-python-deps:
	@echo "Installing Python dependencies..."
	@command -v uv >/dev/null 2>&1 || python -m pip install --user uv
	@test -d .venv || uv venv --python 3.10 && uv sync

train: install-python-deps
	@echo "Training the custom model..."
	uv run python $(TRAIN_SCRIPT) --load-weights --export-model

simulate:
	@rm -rf $(MODEL_DIR)/data/extracted_animals
	uv run python $(MODEL_DIR)/data/simulate_data.py

$(SAVED_MODEL_PATH)/saved_model.pb : train

convert: # $(SAVED_MODEL_PATH)/saved_model.pb
	@echo "Converting the trained model to TensorFlow.js format..."
	# Ensure the output directory exists
	mkdir -p $(TFJS_OUTPUT_DIR)
	TF_ENABLE_ONEDNN_OPTS=0 uv run tensorflowjs_converter \
		--input_format=tf_saved_model \
		--output_node_names='$(OUTPUT_NODE_NAMES)' \
		--output_format=tfjs_graph_model \
		$(SAVED_MODEL_PATH) \
		$(TFJS_OUTPUT_DIR)

model-clean:
	@echo "Cleaning up generated model files..."
	rm -rf $(MODEL_DIR)/saved_model
	rm -rf $(TFJS_OUTPUT_DIR)

.PHONY: gemini
gemini:
	@command -v asciinema >/dev/null 2>&1 || { echo "asciinema is not installed. Installing..."; sudo apt install -y asciinema; }
	@command -v gemini >/dev/null 2>&1 || { echo "gemini is not installed. Installing..."; npm install -g @google/gemini-cli; }
	@echo "Running Gemini model training and conversion..."
	@exec asciinema rec --overwrite --command "gemini -m gemini-2.5-flash" "./.gemini/gemini-$$(date).cast"
