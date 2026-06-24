# WISDM evaluation protocol

Standardization and input/hidden calibration use training windows only. Quality is reported on held-out test subjects. Phase 8C chooses the lowest canonical IDs: two INT8-correct samples per class and up to one INT8 error per class, then validates RTL against integer reference traces rather than ground truth.
