title: Fake News Verifier
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: docker
app_file: server/app.py
pinned: false

🤖 Agentic Fake News Verifier

This project is a custom fact-checking environment built for the Meta PyTorch OpenEnv Hackathon.

It evaluates how effectively an LLM agent can verify news headlines using search and verification actions.
Project Structure

    server/logic.py — Core environment engine with task logic and reward handling
    server/app.py — FastAPI server exposing the OpenEnv-compatible endpoints
    server/tasks.py — Task definitions and helpers
    Dockerfile — Container configuration for Hugging Face Spaces deployment
    pyproject.toml — Project metadata and dependency management
    requirements.txt — Python dependencies
    openenv.yaml — Environment/task definitions

Environment Overview

The environment is designed to test a model’s ability to:

    search for relevant evidence
    reason over the available context
    produce a final verification decision

Reward Logic

Example reward structure used by the environment:

    Search action: +0.15
    Correct verification: +0.95
    Incorrect verification: +0.05
    Default step: +0.05

Local Testing

Install dependencies:

pip install -r requirements.txt
