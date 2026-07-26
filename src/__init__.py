"""The Watchtower — behavioural anomaly detection for cybersecurity.

Implemented:

    data_generator.py     synthetic event-log generator + CLI (Deliverable 1)
    generator/            config, entities, profiles, attacks

Planned (see CLAUDE.md §6):

    profiler.py           rolling per-entity behavioural baseline (drift-aware)
    features.py           sequence feature engineering
    sequence_detector.py  sequence-aware anomaly scorer
    classifier.py         anomaly-type classifier
    explainer.py          analyst-readable explanations
    evaluate.py           precision @ top-1% alert budget, per-class metrics
"""
