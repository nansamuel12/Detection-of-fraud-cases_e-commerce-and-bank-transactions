# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-12-30

### Added
- **Model Explainability**: Integrated SHAP and LIME for global and local model interpretation.
  - Added `src/explainability.py` with `ExplainabilityAnalyzer` class.
  - Implemented automatic generation of feature importance plots and SHAP summaries.
  - Added support for "Representative Case Analysis" (TP/FP/FN).
- **Code Hardening**:
  - Added strict type hints to all source modules (`src/`).
  - Implemented custom exception classes (`DataLoaderError`, `PreprocessingError`, `ModelTrainingError`) for robust error handling.
- **Testing**:
  - Created a `tests/` directory with unit tests for data loading, preprocessing, and training.
  - achieve 100% pass rate with `pytest`.
- **CI/CD**:
  - Added GitHub Actions workflow (`.github/workflows/ci.yml`) to run linting and tests on every push.
- **Documentation**:
  - Added `docs/BRANCHING_STRATEGY.md` defining the development workflow.
  - Added `.github/pull_request_template.md` for standardized code reviews.
  - Added automatic generation of `FINAL_MODEL_REPORT.md` and `business_recommendations.md`.

### Changed
- **Model Training Pipeline**:
  - Enhanced `train_models.py` to support full model comparison and automated selection.
  - Implemented weighted scoring for model selection (Recall vs. Precision vs. Interpretability).
- **Preprocessing**: 
  - Improved `preprocessing.py` to handle potential data leakage and ensure robust type safety.

### Fixed
- Resolved `SettingWithCopyWarning` in data loading.
- Fixed dependency issues with `types-pandas`.
