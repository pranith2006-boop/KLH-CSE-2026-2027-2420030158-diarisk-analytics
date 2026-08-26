# SHAP and LIME Method Comparison

SHAP provides model-aware global importance and additive local feature contributions. TreeExplainer is used here because both Phase 6 selections are tree models. LIME is model-agnostic and fits a local surrogate around one transformed record. Their results can differ because they use different approximation and weighting methodologies; neither is universally superior. Encoded categorical features remain represented as transformed source-category names, which is a limitation when aggregating back to original columns.
