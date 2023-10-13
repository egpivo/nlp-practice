SHELL = /bin/bash
EXECUTABLE:=$(shell poetry env info --path)

rebuild_openai:
	docker-compose up -d --no-deps --build openai

activate:
	source $(EXECUTABLE)/bin/activate
