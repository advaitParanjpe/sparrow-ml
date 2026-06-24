# WISDM data contract

Phase 8 uses phone accelerometer files through `WISDM_ROOT`. Only the four canonical activities and 80-sample, 50%-overlap windows are accepted. Windows are assigned by deterministic subject-level split (seed `20260623`); serialized artifacts contain source basenames and `WISDM_ROOT`, never an absolute dataset path.
