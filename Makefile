SHELL = /bin/bash
.ONESHELL:
.DEFAULT_GOAL: help

help: ## Prints available commands
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

bundle.install: ## Install the Ruby gems
	@docker-compose run ruby bundle install

rubocop: ## Runs Rubocop with auto-correction
	@docker-compose run ruby rubocop -A

run.tests: ## Runs Unit tests
	@docker-compose run ruby ruby -r ./test/test_helper test/all.rb

bash: ## Creates a bash container
	@docker-compose run ruby bash
