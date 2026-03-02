"""
Shared helper modules for the source preference evaluation framework.

These modules provide common functionality used across all experiment scripts:
- config: Constants and configuration values
- models: Data classes for entities, sources, conflicts, and results
- data_loader: Loading data from HuggingFace datasets
- conflict_analyzer: Extracting conflicting field values from entities
- prompt_builder: Constructing prompts with tables and questions
- model_inference: Loading LLMs and extracting answer probabilities
- source_generator: Generating newspaper, government, person, and social media sources
- evaluation: Computing position bias, source preference (SP), and significance tests
"""
