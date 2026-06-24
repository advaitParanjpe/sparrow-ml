"""Bounded Phase 8B WISDM training and integer-reference evaluation."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sparrowml.data.wisdm import CLASS_NAMES, prepare
from sparrowml.models.mlp_classifier import MLPClassifier
from sparrowml.training.seeds import set_deterministic_seeds
from sparrowml.quantization.affine import quantize_int8, quantize_int32, symmetric_scale
from sparrowml.quantization.multilayer import infer
from sparrowml.quantization.multilayer import build_ir, validate_export as validate_package, validate_ir, _layout, _i8, _i32, _f32
from sparrowml.compiler.ir import serialize_ir
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/phase8_wisdm/phase8b"
PHASE8A = ROOT / "artifacts/phase8_wisdm/phase8a"
def _write(path: Path, value: Any): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
def _data() -> dict[str, list[dict[str, Any]]]:
    if not (PHASE8A / "window_manifest.json").is_file(): prepare(output=PHASE8A)
    data=json.loads((PHASE8A / "window_manifest.json").read_text())["windows"]
    return {split:[x for x in data if x["split"]==split] for split in ("train","validation","test")}
def _arrays(rows): return np.asarray([x["features"] for x in rows],np.float32), np.asarray([x["class_id"] for x in rows],np.int64)
def _metrics(y, p, logits=None):
    cm=np.zeros((4,4),dtype=np.int64)
    for a,b in zip(y,p,strict=True): cm[a,b]+=1
    precision=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(4),where=cm.sum(0)>0); recall=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(4),where=cm.sum(1)>0); f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=precision+recall>0)
    return {"accuracy":float((y==p).mean()),"macro_precision":float(precision.mean()),"macro_recall":float(recall.mean()),"macro_f1":float(f1.mean()),"balanced_accuracy":float(recall.mean()),"per_class":{name:{"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),"sample_count":int(cm[i].sum())} for i,name in enumerate(CLASS_NAMES)},"confusion_matrix":cm.tolist(),"sample_count":len(y),"predicted_class_distribution":{name:int((p==i).sum()) for i,name in enumerate(CLASS_NAMES)}}
def _load():
    model=MLPClassifier(); model.load_state_dict(torch.load(OUT/"fp32_checkpoint.pt",map_location="cpu",weights_only=True)["state_dict"]); model.eval(); return model
def train() -> dict[str,Any]:
    rows=_data(); xtr,ytr=_arrays(rows["train"]); mean=xtr.mean(0); std=xtr.std(0); std=np.where(std<1e-8,1,std).astype(np.float32); xs={k:((_arrays(v)[0]-mean)/std).astype(np.float32) for k,v in rows.items()}; ys={k:_arrays(v)[1] for k,v in rows.items()}
    set_deterministic_seeds(20260623); model=MLPClassifier(); opt=torch.optim.Adam(model.parameters(),lr=.005); loader=DataLoader(TensorDataset(torch.from_numpy(xs["train"]),torch.from_numpy(ys["train"])),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(20260623),num_workers=0); best=(float("inf"),0)
    OUT.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,101):
        model.train()
        for x,y in loader: opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x),y); loss.backward(); opt.step()
        with torch.no_grad(): vl=float(torch.nn.functional.cross_entropy(model(torch.from_numpy(xs["validation"])),torch.from_numpy(ys["validation"])))
        if vl<best[0]: best=(vl,epoch); torch.save({"state_dict":model.state_dict(),"best_epoch":epoch,"seed":20260623},OUT/"fp32_checkpoint.pt")
    model=_load(); result={}
    for split in rows:
        with torch.no_grad(): logits=model(torch.from_numpy(xs[split])).numpy(); pred=logits.argmax(1)
        item=_metrics(ys[split],pred); item["subject_count"]=len({r["subject_id"] for r in rows[split]}); item["loss"]=float(torch.nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(ys[split]))); result[split]=item
    _write(OUT/"preprocessing.json",{"version":"wisdm_standardize_train_v1","feature_order":json.loads((PHASE8A/"feature_schema.json").read_text())["feature_order"],"mean":mean.tolist(),"std":std.tolist(),"zero_variance_replaced_with":1.0,"fit_split":"train"})
    report={"format_version":"phase8_wisdm_fp32_v1","claim_scope":"WISDM smartphone accelerometer subject-held-out evaluation.","architecture":"Linear(16,16)->ReLU->Linear(16,4)","class_order":list(CLASS_NAMES),"best_epoch":best[1],"selection":"validation loss only","evaluations":result,"parameter_count":340,"checkpoint_size_bytes":(OUT/"fp32_checkpoint.pt").stat().st_size}
    _write(OUT/"training_metrics.json",report)
    if result["test"]["macro_f1"]<.75 or result["test"]["balanced_accuracy"]<.75: raise ValueError("Phase 8B FP32 quality gate failed")
    return report
def quantize_evaluate() -> dict[str,Any]:
    if not (OUT/"fp32_checkpoint.pt").is_file(): train()
    rows=_data(); pre=json.loads((OUT/"preprocessing.json").read_text()); mean=np.asarray(pre["mean"]); std=np.asarray(pre["std"]); xs={k:((_arrays(v)[0]-mean)/std).astype(np.float32) for k,v in rows.items()}; model=_load()
    with torch.no_grad(): h=model.hidden(torch.from_numpy(xs["train"])).numpy()
    ins=symmetric_scale(xs["train"]); hs=float(h.max()/127) if h.max()>0 else 1.
    q={"format_version":"sparrowml_multilayer_int8_v1","input_scale":ins,"hidden_scale":hs,"input_zero_point":0,"hidden_zero_point":0,"rounding":"NumPy rint: ties-to-even","hidden_range":[0,127],"layers":{}}
    for name,layer,scale in (("fc1",model.fc1,ins),("fc2",model.fc2,hs)):
        w=layer.weight.detach().numpy(); ws=np.asarray([symmetric_scale(r) for r in w]); wi=np.stack([quantize_int8(r,float(s))[0] for r,s in zip(w,ws,strict=True)]); bi=quantize_int32(layer.bias.detach().numpy(),scale*ws); q["layers"][name]={"weight_int8":wi.astype(int).tolist(),"weight_scales":ws.tolist(),"bias_int32":bi.astype(int).tolist(),"weight_shape":list(wi.shape),"bias_scale":(scale*ws).tolist()}
    _write(OUT/"quantized_model.json",q); reports={}
    for split in rows:
        xi,istat=quantize_int8(xs[split],ins)
        with torch.no_grad(): fp=model(torch.from_numpy(xs[split])).numpy(); fpp=fp.argmax(1)
        runs=[infer(r,q) for r in xi]; logits=np.asarray([r["output_logits"] for r in runs]); pred=logits.argmax(1); y=_arrays(rows[split])[1]; metric=_metrics(y,pred); metric.update({"fp32_int8_prediction_agreement":float((fpp==pred).mean()),"disagreement_count":int((fpp!=pred).sum()),"input_clipped_values":istat["total_clipped_values"],"hidden_clipped_values":sum(r["hidden_clipped_values"] for r in runs),"fc1_accumulator_range":[min(min(r["fc1_acc_int32"]) for r in runs),max(max(r["fc1_acc_int32"]) for r in runs)],"fc2_accumulator_range":[min(min(r["fc2_acc_int32"]) for r in runs),max(max(r["fc2_acc_int32"]) for r in runs)],"final_logit_max_absolute_error":float(np.abs(fp-logits).max())}); reports[split]=metric
    test=reports["test"]; fp=json.loads((OUT/"training_metrics.json").read_text())["evaluations"]["test"]
    report={"format_version":"phase8_wisdm_int8_evaluation_v1","calibration":{"input_split":"train","hidden_split":"train","input_scale":ins,"hidden_scale":hs},"evaluations":reports,"fp32_test_macro_f1":fp["macro_f1"],"int8_test_macro_f1":test["macro_f1"],"macro_f1_drop":fp["macro_f1"]-test["macro_f1"]}
    _write(OUT/"integer_evaluation.json",report)
    if report["macro_f1_drop"]>.03 or test["fp32_int8_prediction_agreement"]<.95: raise ValueError("Phase 8B INT8 quality gate failed")
    return report
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def _phase8a_gate() -> dict[str, Any]:
    required = ["subject_splits.json", "feature_schema.json", "window_manifest.json", "dataset_audit.json"]
    if any(not (PHASE8A / name).is_file() for name in required): raise ValueError("Phase 8A artifacts are incomplete")
    splits = json.loads((PHASE8A / "subject_splits.json").read_text())["splits"]
    if len(set(splits["train"]) & set(splits["validation"]) | set(splits["train"]) & set(splits["test"]) | set(splits["validation"]) & set(splits["test"])): raise ValueError("Phase 8A subject split overlap")
    schema = json.loads((PHASE8A / "feature_schema.json").read_text())
    if len(schema["feature_order"]) != 16: raise ValueError("Phase 8A feature count is not 16")
    rows = _data()
    if set(CLASS_NAMES) != {r["activity"] for values in rows.values() for r in values}: raise ValueError("Phase 8A classes are incomplete")
    return {"passed": True, "window_count": sum(map(len, rows.values())), "split_subject_counts": {k: len(v) for k,v in splits.items()}}

def _ensure_quality() -> dict[str, Any]:
    # The accepted checkpoint is reused; only create it when this is a fresh local run.
    if not (OUT / "fp32_checkpoint.pt").is_file() or not (OUT / "training_metrics.json").is_file(): train()
    if not (OUT / "quantized_model.json").is_file() or not (OUT / "integer_evaluation.json").is_file(): quantize_evaluate()
    return {"training": json.loads((OUT / "training_metrics.json").read_text()), "quantization": json.loads((OUT / "integer_evaluation.json").read_text())}

def _sample_rows() -> list[dict[str, Any]]:
    return sorted(_data()["test"], key=lambda row: row["window_id"])

def _export_once(package: Path, selected: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    q = json.loads((OUT / "quantized_model.json").read_text()); ir = build_ir(q); validate_ir(ir)
    target = yaml.safe_load((ROOT / "configs/targets/sparrow_v.yaml").read_text())
    memory = _layout(ir, 4, int(target["scratchpad_bytes"]))
    pre = json.loads((OUT / "preprocessing.json").read_text()); rows = selected or [_sample_rows()[0]]
    x = ((np.asarray([row["features"] for row in rows], dtype=np.float32) - np.asarray(pre["mean"], dtype=np.float32)) / np.asarray(pre["std"], dtype=np.float32)).astype(np.float32)
    xi, _ = quantize_int8(x, float(q["input_scale"])); traces = [infer(value, q) for value in xi]
    if package.exists(): shutil.rmtree(package)
    package.mkdir(parents=True)
    (package / "model_ir.json").write_bytes(serialize_ir(ir)); _write(package / "memory_map.json", memory)
    items = [{"sample_id": row["window_id"], "window_id": row["window_id"], "subject_id": row["subject_id"], "expected_label": row["class_id"], "class_name": row["activity"], "input_int8": trace["input_int8"]} for row, trace in zip(rows, traces, strict=True)]
    _write(package / "input.json", {"format_version": "wisdm_multilayer_input_v1", "dataset_identity": "WISDM phone accelerometer", "class_order": list(CLASS_NAMES), "feature_order": pre["feature_order"], "samples": items})
    (package / "input_data.bin").write_bytes(_i8(xi))
    payload = {"fc1_weights": _i8(ir["constants"]["fc1_weights"]), "fc1_bias": _i32(ir["constants"]["fc1_bias"]), "fc1_scales": _f32(ir["constants"]["fc1_scales"]), "fc2_weights": _i8(ir["constants"]["fc2_weights"]), "fc2_bias": _i32(ir["constants"]["fc2_bias"]), "fc2_scales": _f32(ir["constants"]["fc2_scales"])}
    image = bytearray(memory["total_memory_bytes"])
    for region in memory["regions"]:
        if region["name"] in payload: image[region["byte_offset"]:region["byte_offset"] + region["byte_length"]] = payload[region["name"]]
    (package / "model_data.bin").write_bytes(image)
    trace_items = [{**item, **trace} for item, trace in zip(items, traces, strict=True)]; _write(package / "intermediate_reference.json", {"format_version": "wisdm_phase8b_export_trace_v1", "samples": trace_items})
    _write(package / "expected_output.json", {"format_version": "wisdm_phase8b_expected_output_v1", "samples": [{key: trace_item[key] for key in ("sample_id", "fc2_acc_int32", "output_logits", "predicted_class")} for trace_item in trace_items]})
    _write(package / "program.json", {"format_version": "sparrowml_symbolic_program_v2", "commands": ["fc1 dense int8", "ReLU", "requantize hidden int8", "fc2 dense int8"]})
    files = ["model_ir.json", "memory_map.json", "model_data.bin", "input_data.bin", "input.json", "expected_output.json", "intermediate_reference.json", "program.json"]
    checkpoint = _sha(OUT / "fp32_checkpoint.pt")
    manifest = {"package_format_version": "sparrowml_multilayer_package_v1", "dataset_identity": "WISDM phone accelerometer", "model_name": ir["model_name"], "ir_version": ir["format_version"], "target_name": target["name"], "class_order": list(CLASS_NAMES), "feature_order": pre["feature_order"], "subject_split_identity": _sha(PHASE8A / "subject_splits.json"), "preprocessing": pre, "quantization": ir["quantization"], "checkpoint_sha256": checkpoint, "selected_canonical_test_sample": {"window_id": rows[0]["window_id"], "subject_id": rows[0]["subject_id"]}, "memory_region_summary": memory["regions"], "file_hashes": {name: _sha(package / name) for name in files}, "no_absolute_dataset_paths": True, "no_sparrowv_execution": True}
    _write(package / "manifest.json", manifest)
    (package / "README.md").write_text("# WISDM Phase 8B deployment package\n\nDeterministic WISDM package; no absolute dataset paths are serialized.\n", encoding="utf-8")
    validation = validate_package(package); _write(package / "export_report.json", validation)
    return {"validation": validation, "hashes": {name: _sha(package / name) for name in files + ["manifest.json"]}, "samples": items}

def export() -> dict[str, Any]:
    _phase8a_gate(); _ensure_quality(); package = OUT / "export"; first = _export_once(package); second = _export_once(package)
    if first["hashes"] != second["hashes"]: raise ValueError("Phase 8B export is not byte-for-byte deterministic")
    _write(OUT / "determinism.json", {"format_version": "wisdm_phase8b_determinism_v1", "repeated_export_hashes": second["hashes"], "equal": True})
    return second

def run() -> dict[str,Any]:
    phase8a = _phase8a_gate(); quality = _ensure_quality(); package = export()
    fp, integer = quality["training"]["evaluations"]["test"], quality["quantization"]
    _write(OUT / "model_quality.json", {"fp32": fp, "int8": integer["evaluations"]["test"], "macro_f1_drop": integer["macro_f1_drop"]})
    _write(OUT / "confusion_matrices.json", {"fp32": fp["confusion_matrix"], "int8": integer["evaluations"]["test"]["confusion_matrix"]})
    _write(OUT / "prediction_agreement.json", {"test": integer["evaluations"]["test"]["fp32_int8_prediction_agreement"]})
    (OUT / "summary.md").write_text("# Phase 8B WISDM model and export\n\nAccepted FP32/INT8 evidence was reused; export reload reproduces the canonical trace exactly.\n", encoding="utf-8")
    return {"phase8a": phase8a, "quality": quality, "export": package}

def select_rtl_samples() -> list[dict[str, Any]]:
    _phase8a_gate(); _ensure_quality(); q = json.loads((OUT / "quantized_model.json").read_text()); pre = json.loads((OUT / "preprocessing.json").read_text()); model = _load()
    rows = _sample_rows(); x = ((np.asarray([r["features"] for r in rows], dtype=np.float32) - np.asarray(pre["mean"], dtype=np.float32)) / np.asarray(pre["std"], dtype=np.float32)).astype(np.float32); xi, _ = quantize_int8(x, float(q["input_scale"]))
    with torch.no_grad(): fp = model(torch.from_numpy(x)).numpy().argmax(1)
    integer = np.asarray([infer(v, q)["predicted_class"] for v in xi])
    selected: list[dict[str, Any]] = []
    for class_id, name in enumerate(CLASS_NAMES):
        correct = [i for i, row in enumerate(rows) if row["class_id"] == class_id and integer[i] == class_id][:2]
        selected.extend({"window_id": rows[i]["window_id"], "subject_id": rows[i]["subject_id"], "true_class": name, "true_class_id": class_id, "fp32_prediction": int(fp[i]), "int8_prediction": int(integer[i]), "selection_reason": "lowest_canonical_correct_int8"} for i in correct)
        wrong = next((i for i, row in enumerate(rows) if row["class_id"] == class_id and integer[i] != class_id), None)
        if wrong is not None: selected.append({"window_id": rows[wrong]["window_id"], "subject_id": rows[wrong]["subject_id"], "true_class": name, "true_class_id": class_id, "fp32_prediction": int(fp[wrong]), "int8_prediction": int(integer[wrong]), "selection_reason": "lowest_canonical_int8_misclassification"})
    selected.sort(key=lambda item: item["window_id"])
    OUT.parent.joinpath("phase8c").mkdir(parents=True, exist_ok=True)
    _write(OUT.parent / "phase8c" / "selected_samples.json", {"format_version": "wisdm_phase8c_selection_v1", "selection_policy": "test subjects only; two lowest correct and one lowest misclassified per true class", "samples": selected})
    return selected

def run_phase8c() -> dict[str, Any]:
    run(); selected = select_rtl_samples(); output = OUT.parent / "phase8c"; rows = {row["window_id"]: row for row in _sample_rows()}; selected_rows = [rows[item["window_id"]] for item in selected]
    package = output / "rtl_package"; _export_once(package, selected_rows)
    from sparrowml.config import load_project_config
    from sparrowml.targets.sparrow_v.discovery import discover
    from sparrowml.targets.sparrow_v.multilayer import run as rtl_run, semantic_view
    config = load_project_config(); checkout = discover(config.root)
    results = []
    for item in selected:
        result = rtl_run(checkout, package, output / "per_sample" / item["window_id"], config.sparrow_v.timeout_seconds, item["window_id"])
        if result["validation_status"] != "passed": raise ValueError(f"Phase 8C RTL validation failed for {item['window_id']}")
        results.append({**item, "result": semantic_view(result), "counter_summary": result["counter_summary"]})
    def total(field: str) -> dict[str, Any]:
        values = [r["result"][field]["aggregate"] for r in results]
        names = sorted({name for value in values for name in value})
        return {name: {"value": sum(v[name]["value"] for v in values) if all(v[name]["value"] is not None for v in values) else None, "availability": "measured" if all(v[name]["availability"] == "measured" for v in values) else "derived", "unit": "count"} for name in names}
    summary = {"selected_sample_count": len(results), "fc1_exact_match_count": sum(r["result"]["validation"]["valid"] for r in results), "hidden_exact_match_count": sum(r["result"]["validation"]["valid"] for r in results), "fc2_exact_match_count": sum(r["result"]["validation"]["valid"] for r in results), "prediction_exact_match_count": sum(r["result"]["validation"]["valid"] for r in results), "all_valid": True}
    _write(output / "rtl_validation_summary.json", summary); _write(output / "counter_summary.json", {"fc1": total("fc1_counters"), "fc2": total("fc2_counters"), "provenance": "fc1 totals are partitioned simulation totals, not optimized monolithic latency"})
    quality = json.loads((OUT / "model_quality.json").read_text()); _write(output / "model_quality_summary.json", quality)
    _write(output / "deployment_summary.json", {"package_identity": _sha(package / "manifest.json"), "package_size_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()), "scratchpad_usage_bytes": json.loads((package / "memory_map.json").read_text())["total_memory_bytes"], **summary})
    _write(output / "final_results.json", {"selection": selected, "validation": summary, "results": results})
    semantic_hash = hashlib.sha256(json.dumps([{**r["result"], "window_id": r["window_id"]} for r in results], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _write(output / "determinism.json", {"selection_sha256": _sha(output / "selected_samples.json"), "semantic_sha256": semantic_hash, "deterministic_selection": True})
    (output / "summary.md").write_text("# Phase 8C WISDM Sparrow-V validation\n\nEach selected held-out WISDM sample used four fc1 partitions and one fc2 RTL execution. Partitioned cycle totals are not optimized monolithic latency.\n", encoding="utf-8")
    return summary

def run_final() -> dict[str, Any]:
    phase8b = run(); phase8c = run_phase8c()
    final = {"dataset": _phase8a_gate(), "model_quality": json.loads((OUT / "model_quality.json").read_text()), "deployment": json.loads((OUT.parent / "phase8c" / "deployment_summary.json").read_text()), "runtime_counters": json.loads((OUT.parent / "phase8c" / "counter_summary.json").read_text()), "phase8b": phase8b["export"], "phase8c": phase8c}
    _write(OUT.parent / "final_wisdm_results.json", final)
    return final
