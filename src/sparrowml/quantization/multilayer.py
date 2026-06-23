"""Deterministic, deliberately fixed Phase 6 MLP training and export workflow."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from sparrowml.data.fixture import CLASS_NAMES, generate_fixture, load_fixture, write_fixture
from sparrowml.data.preprocessing import Standardization, fit_standardization
from sparrowml.models.mlp_classifier import MLPClassifier
from sparrowml.training.seeds import set_deterministic_seeds
from sparrowml.training.trainer import split_examples
from .affine import INT32_MAX, INT32_MIN, quantize_int8, quantize_int32, symmetric_scale
from sparrowml.compiler.ir import serialize_ir

IR_VERSION = "sparrowml_ir_v2_multilayer"


def _root() -> Path: return Path(__file__).resolve().parents[3]
def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    root = _root(); file = root / (path or "configs/experiments/multilayer_int8_baseline.yaml")
    try:
        value = yaml.safe_load(file.read_text()); assert value["training"]["epochs"] >= 1
        assert value["alignment"] == 4 and value["hidden_quantization"]["range"] == [0, 127]
    except (OSError, KeyError, TypeError, AssertionError, yaml.YAMLError) as exc: raise ValueError(f"invalid Phase 6 configuration {file}: {exc}") from exc
    return value, root
def _paths(c: dict[str, Any], root: Path) -> tuple[Path, Path]: return root / c["dataset"]["fixture_directory"], root / c["output_directory"]
def _ensure_fixture(c: dict[str, Any], root: Path) -> Path:
    fixture, _ = _paths(c, root)
    if not (fixture / "examples.jsonl").is_file():
        examples = generate_fixture(seed=20260623, split_seed=20260623, samples_per_class=128, split_counts_per_class={"train": 90, "validation": 19, "test": 19})
        write_fixture(fixture, examples, 20260623, 20260623)
    return fixture
def _matrix(examples: list[Any]) -> np.ndarray: return np.asarray([e.features for e in examples], dtype=np.float32)
def _labels(examples: list[Any]) -> np.ndarray: return np.asarray([e.class_id for e in examples], dtype=np.int64)
def _eval(model: MLPClassifier, x: np.ndarray, examples: list[Any]) -> dict[str, Any]:
    with torch.no_grad(): logits = model(torch.from_numpy(x)).numpy(); loss = float(torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(_labels(examples))).item())
    pred = logits.argmax(1); labels = _labels(examples); cm = np.zeros((4, 4), dtype=np.int64)
    for y, p in zip(labels, pred, strict=True): cm[y, p] += 1
    return {"loss": loss, "fixture_accuracy": float((pred == labels).mean()), "confusion_matrix": cm.tolist()}
def train(config_path: str | Path | None = None) -> dict[str, Any]:
    c, root = _config(config_path); fixture = _ensure_fixture(c, root); output = _paths(c, root)[1]; splits = split_examples(load_fixture(fixture)); set_deterministic_seeds(int(c["seed"])); norm = fit_standardization(splits["train"]); matrices = {k: norm.transform(v) for k,v in splits.items()}
    model = MLPClassifier(); opt = torch.optim.Adam(model.parameters(), lr=float(c["training"]["learning_rate"])); loader = DataLoader(TensorDataset(torch.from_numpy(matrices["train"]), torch.from_numpy(_labels(splits["train"]))), batch_size=int(c["training"]["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(int(c["training"]["dataloader_seed"])), num_workers=0)
    output.mkdir(parents=True, exist_ok=True); checkpoint = output / "fp32_checkpoint.pt"; best_loss = float("inf"); best_epoch = 0
    for epoch in range(1, int(c["training"]["epochs"]) + 1):
        model.train()
        for x, y in loader: opt.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x), y); loss.backward(); opt.step()
        validation = _eval(model, matrices["validation"], splits["validation"])
        if validation["loss"] < best_loss: best_loss, best_epoch = validation["loss"], epoch; torch.save({"state_dict": model.state_dict(), "best_epoch": epoch, "seed": c["seed"]}, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True); model.load_state_dict(payload["state_dict"]); evaluations = {k: _eval(model, matrices[k], splits[k]) for k in splits}
    if evaluations["test"]["fixture_accuracy"] < c["acceptance_gates"]["minimum_test_fixture_accuracy"]: raise ValueError("FP32 test quality gate failed")
    metrics = {"experiment_version": "phase6_mlp_fp32_v1", "claim_scope": "Synthetic fixture accuracy only.", "architecture": "Linear(16,16) -> ReLU -> Linear(16,4)", "parameter_count": 340, "per_layer_parameter_count": {"fc1": 272, "fc2": 68}, "best_epoch": best_epoch, "training": c["training"], "seeds": {"python_numpy_torch": c["seed"], "dataloader": c["training"]["dataloader_seed"]}, "evaluations": evaluations, "checkpoint_size_bytes": checkpoint.stat().st_size}
    _json(output / "training_metrics.json", metrics); _json(output / "preprocessing.json", norm.as_dict()); return metrics
def _load_model(output: Path) -> MLPClassifier:
    model = MLPClassifier(); model.load_state_dict(torch.load(output / "fp32_checkpoint.pt", map_location="cpu", weights_only=True)["state_dict"]); model.eval(); return model
def calibrate(config_path: str | Path | None = None) -> dict[str, Any]:
    c, root = _config(config_path); fixture, output = _paths(c, root)
    if not (output / "fp32_checkpoint.pt").is_file(): train(config_path)
    splits = split_examples(load_fixture(fixture)); pre = json.loads((output / "preprocessing.json").read_text()); norm = Standardization(np.asarray(pre["mean"]), np.asarray(pre["std"]), pre["version"]); x = norm.transform(splits["train"]); model = _load_model(output)
    with torch.no_grad(): hidden = model.hidden(torch.from_numpy(x)).numpy()
    in_scale = symmetric_scale(x); hmax = float(hidden.max()); hidden_scale = hmax / 127 if hmax else 1.0
    iq, istat = quantize_int8(x, in_scale); hq_raw = np.rint(hidden / hidden_scale).astype(np.int64); hq = np.clip(hq_raw, 0, 127).astype(np.int8)
    input_report = {"calibration_split": "train", "calibration_sample_count": len(x), "input_scale": in_scale, "zero_point": 0, "minimum": float(x.min()), "maximum": float(x.max()), "maximum_absolute": float(np.abs(x).max()), "quantization": istat}
    hidden_report = {"calibration_split": "train", "calibration_sample_count": len(x), "minimum": float(hidden.min()), "maximum": hmax, "maximum_absolute": float(np.abs(hidden).max()), "hidden_scale": hidden_scale, "zero_point": 0, "clipped_values": int((hq_raw != hq).sum()), "clipping_percentage": float((hq_raw != hq).mean() * 100), "valid_code_range": [0, 127]}
    _json(output / "input_calibration.json", input_report); _json(output / "hidden_calibration.json", hidden_report); return {"input": input_report, "hidden": hidden_report}
def quantize(config_path: str | Path | None = None) -> dict[str, Any]:
    c, root = _config(config_path); _, output = _paths(c, root); calibration = calibrate(config_path); m = _load_model(output)
    result: dict[str, Any] = {"format_version": "sparrowml_multilayer_int8_v1", "model_name": "Linear(16,16)->ReLU->Linear(16,4)", "input_scale": calibration["input"]["input_scale"], "hidden_scale": calibration["hidden"]["hidden_scale"], "input_zero_point": 0, "hidden_zero_point": 0, "rounding": "NumPy rint: ties-to-even", "hidden_range": [0,127], "layers": {}}
    for name, layer, activation_scale in (("fc1", m.fc1, result["input_scale"]), ("fc2", m.fc2, result["hidden_scale"])):
        w = layer.weight.detach().numpy(); scales = np.asarray([symmetric_scale(row) for row in w]); wi = np.stack([quantize_int8(row, float(scale))[0] for row, scale in zip(w, scales, strict=True)]); bi = quantize_int32(layer.bias.detach().numpy(), activation_scale * scales)
        result["layers"][name] = {"weight_int8": wi.astype(int).tolist(), "weight_scales": scales.tolist(), "weight_zero_points": [0]*len(scales), "bias_int32": bi.astype(int).tolist(), "weight_shape": list(wi.shape), "bias_scale": (activation_scale*scales).tolist()}
    _json(output / "quantized_model.json", result); return result
def infer(input_int8: np.ndarray, q: dict[str, Any]) -> dict[str, Any]:
    a = np.asarray(input_int8, dtype=np.int8).astype(np.int64); l1,l2=q["layers"]["fc1"],q["layers"]["fc2"]; w1=np.asarray(l1["weight_int8"],dtype=np.int8).astype(np.int64); b1=np.asarray(l1["bias_int32"],dtype=np.int32).astype(np.int64); acc1=b1+w1@a
    if np.any((acc1 < INT32_MIN)|(acc1 > INT32_MAX)): raise ValueError("fc1 accumulator exceeds INT32")
    pre=acc1.astype(np.float64)*float(q["input_scale"])*np.asarray(l1["weight_scales"],dtype=np.float64); relu=np.maximum(pre,0); raw=np.rint(relu/float(q["hidden_scale"])).astype(np.int64); hidden=np.clip(raw,0,127).astype(np.int8)
    w2=np.asarray(l2["weight_int8"],dtype=np.int8).astype(np.int64); b2=np.asarray(l2["bias_int32"],dtype=np.int32).astype(np.int64); acc2=b2+w2@hidden.astype(np.int64)
    if np.any((acc2 < INT32_MIN)|(acc2 > INT32_MAX)): raise ValueError("fc2 accumulator exceeds INT32")
    logits=acc2.astype(np.float64)*float(q["hidden_scale"])*np.asarray(l2["weight_scales"],dtype=np.float64)
    return {"input_int8": a.astype(int).tolist(),"fc1_acc_int32":acc1.astype(int).tolist(),"hidden_pre_relu_real":pre.tolist(),"hidden_relu_real":relu.tolist(),"hidden_int8":hidden.astype(int).tolist(),"fc2_acc_int32":acc2.astype(int).tolist(),"output_logits":logits.tolist(),"predicted_class":int(logits.argmax()),"hidden_clipped_values":int((raw!=hidden).sum())}
def evaluate(config_path: str | Path | None = None) -> dict[str, Any]:
    c,root=_config(config_path); fixture,output=_paths(c,root); q=quantize(config_path); splits=split_examples(load_fixture(fixture)); pre=json.loads((output/"preprocessing.json").read_text()); norm=Standardization(np.asarray(pre["mean"]),np.asarray(pre["std"]),pre["version"]); m=_load_model(output); reports={}; traces=[]
    for name, examples in splits.items():
        x=norm.transform(examples); xi,_=quantize_int8(x,float(q["input_scale"]))
        with torch.no_grad(): fp=m(torch.from_numpy(x)).numpy().astype(np.float64)
        runs=[infer(row,q) for row in xi]; logits=np.asarray([r["output_logits"] for r in runs]); p=logits.argmax(1); fp_p=fp.argmax(1); y=_labels(examples); cm=np.zeros((4,4),dtype=np.int64)
        for target,pred in zip(y,p,strict=True):cm[target,pred]+=1
        reports[name]={"fp32_fixture_accuracy":float((fp_p==y).mean()),"int8_fixture_accuracy":float((p==y).mean()),"prediction_agreement":float((p==fp_p).mean()),"disagreement_count":int((p!=fp_p).sum()),"confusion_matrix":cm.tolist(),"final_logit_max_absolute_error":float(np.abs(fp-logits).max()),"final_logit_mean_absolute_error":float(np.abs(fp-logits).mean()),"final_logit_rms_error":float(np.sqrt(np.mean((fp-logits)**2))),"hidden_activation_clipped_values":sum(r["hidden_clipped_values"] for r in runs),"fc1_accumulator_range":[min(min(r["fc1_acc_int32"]) for r in runs),max(max(r["fc1_acc_int32"]) for r in runs)],"fc2_accumulator_range":[min(min(r["fc2_acc_int32"]) for r in runs),max(max(r["fc2_acc_int32"]) for r in runs)]}
        traces.extend([{"sample_id":e.sample_id,"expected_label":e.class_id,**r} for e,r in zip(examples,runs,strict=True)])
    test=reports["test"]; gates=c["acceptance_gates"]
    if test["int8_fixture_accuracy"]<gates["minimum_test_fixture_accuracy"] or test["fp32_fixture_accuracy"]-test["int8_fixture_accuracy"]>gates["maximum_test_accuracy_drop"] or test["prediction_agreement"]<gates["minimum_prediction_agreement"]: raise ValueError("Phase 6 INT8 quality gate failed")
    result={"format_version":"phase6_integer_evaluation_v1","claim_scope":"Synthetic fixture accuracy only.","evaluations":reports}; _json(output/"integer_evaluation.json",result); _json(output/"intermediate_traces.json",{"format_version":"phase6_intermediate_traces_v1","traces":traces}); return result

def _tensor(name: str, shape: list[int], element_type: str, role: str, domain: str) -> dict[str, Any]:
    size = {"int8":1,"int32":4,"float32":4}[element_type]
    for d in shape: size *= d
    return {"name":name,"shape":shape,"element_type":element_type,"role":role,"quantization_domain":domain,"byte_size":size}
def build_ir(q: dict[str, Any]) -> dict[str, Any]:
    tensors=[_tensor("input_int8",[16],"int8","input","input"),_tensor("fc1_weights",[16,16],"int8","constant","fc1_weight"),_tensor("fc1_bias",[16],"int32","constant","fc1_bias"),_tensor("fc1_scales",[16],"float32","constant","fc1_weight_scale"),_tensor("fc1_acc_int32",[16],"int32","intermediate","fc1_accumulator"),_tensor("hidden_relu_real",[16],"float32","logical_intermediate","hidden_real"),_tensor("hidden_int8",[16],"int8","intermediate","hidden_int8"),_tensor("fc2_weights",[4,16],"int8","constant","fc2_weight"),_tensor("fc2_bias",[4],"int32","constant","fc2_bias"),_tensor("fc2_scales",[4],"float32","constant","fc2_weight_scale"),_tensor("fc2_acc_int32",[4],"int32","intermediate","fc2_accumulator"),_tensor("output_logits",[4],"float32","output","output_logits")]
    ops=[{"op_type":"DenseLinearInt8","input":"input_int8","output":"fc1_acc_int32","weights":"fc1_weights","bias":"fc1_bias","weight_scales":"fc1_scales","feature_count":16,"output_count":16,"accumulation_type":"int32"},{"op_type":"ReLU","input":"fc1_acc_int32","output":"hidden_relu_real","threshold_semantics":"max(0, acc*input_scale*per_channel_weight_scale)","element_count":16},{"op_type":"RequantizeInt8","input":"hidden_relu_real","output":"hidden_int8","input_scales":q["layers"]["fc1"]["weight_scales"],"output_scale":q["hidden_scale"],"output_zero_point":0,"rounding":"NumPy rint ties-to-even","clamp_min":0,"clamp_max":127},{"op_type":"DenseLinearInt8","input":"hidden_int8","output":"fc2_acc_int32","weights":"fc2_weights","bias":"fc2_bias","weight_scales":"fc2_scales","feature_count":16,"output_count":4,"accumulation_type":"int32"}]
    return {"format_version":IR_VERSION,"model_name":"Linear(16,16)->ReLU->Linear(16,4)","input_tensor":"input_int8","output_tensor":"output_logits","tensors":tensors,"operators":ops,"constants":{"fc1_weights":q["layers"]["fc1"]["weight_int8"],"fc1_bias":q["layers"]["fc1"]["bias_int32"],"fc1_scales":q["layers"]["fc1"]["weight_scales"],"fc2_weights":q["layers"]["fc2"]["weight_int8"],"fc2_bias":q["layers"]["fc2"]["bias_int32"],"fc2_scales":q["layers"]["fc2"]["weight_scales"]},"quantization":{"input_scale":q["input_scale"],"hidden_scale":q["hidden_scale"],"zero_points":{"input":0,"hidden":0},"rounding":"NumPy rint ties-to-even","hidden_range":[0,127]}}
def validate_ir(ir: dict[str, Any]) -> None:
    if ir.get("format_version") != IR_VERSION or [x.get("op_type") for x in ir.get("operators",[])] != ["DenseLinearInt8","ReLU","RequantizeInt8","DenseLinearInt8"]: raise ValueError("invalid Phase 6 operator sequence")
    tensors={x["name"]:x for x in ir.get("tensors",[])}
    if len(tensors)!=len(ir.get("tensors",[])) or tensors.get("input_int8",{}).get("shape") != [16] or tensors.get("output_logits",{}).get("shape") != [4]: raise ValueError("invalid Phase 6 tensors")
    producers={"input_int8":None}
    for op in ir["operators"]:
        if op["input"] not in tensors or op["output"] not in tensors or op["output"] in producers: raise ValueError("invalid producer/consumer graph")
        producers[op["output"]]=op
    if ir["operators"][-1]["output"] != "fc2_acc_int32" or ir["operators"][2]["clamp_min"] != 0 or ir["operators"][2]["clamp_max"] != 127: raise ValueError("invalid Phase 6 graph semantics")
def _layout(ir: dict[str, Any], alignment: int, capacity: int) -> dict[str, Any]:
    names=[("input", "input_int8"),("fc1_weights","fc1_weights"),("fc1_bias","fc1_bias"),("fc1_scales","fc1_scales"),("hidden","hidden_int8"),("fc2_weights","fc2_weights"),("fc2_bias","fc2_bias"),("fc2_scales","fc2_scales"),("output","fc2_acc_int32")]; tensors={x["name"]:x for x in ir["tensors"]}; offset=0; regions=[]
    for region, tensor in names:
        offset=(offset+alignment-1)//alignment*alignment; item=tensors[tensor]; regions.append({"name":region,"source_tensor":tensor,"byte_offset":offset,"byte_length":item["byte_size"],"alignment":alignment,"element_type":item["element_type"],"logical_shape":item["shape"]}); offset+=item["byte_size"]
    total=(offset+alignment-1)//alignment*alignment
    if total>capacity: raise ValueError("Phase 6 package exceeds scratchpad capacity")
    return {"format_version":"sparrowml_memory_map_v2","regions":regions,"total_memory_bytes":total,"target_capacity_bytes":capacity,"byte_order":"little-endian"}
def _i8(v: Any) -> bytes: return np.asarray(v,dtype=np.int8).tobytes()
def _i32(v: Any) -> bytes: return np.asarray(v,dtype="<i4").tobytes()
def _f32(v: Any) -> bytes: return np.asarray(v,dtype="<f4").tobytes()
def export(config_path: str | Path | None = None) -> Path:
    c,root=_config(config_path); fixture,output=_paths(c,root); result=evaluate(config_path); q=json.loads((output/"quantized_model.json").read_text()); ir=build_ir(q); validate_ir(ir); package=output/"export"
    if package.exists(): shutil.rmtree(package)
    package.mkdir(parents=True); target=yaml.safe_load((root/c["target"]).read_text()); memory=_layout(ir,int(c["alignment"]),int(target["scratchpad_bytes"])); (package/"model_ir.json").write_bytes(serialize_ir(ir)); _json(package/"memory_map.json",memory)
    examples={e.sample_id:e for e in load_fixture(fixture)}; selected=[examples[x] for x in [e.sample_id for e in split_examples(list(examples.values()))["test"][:4]]]; pre=json.loads((output/"preprocessing.json").read_text()); norm=Standardization(np.asarray(pre["mean"]),np.asarray(pre["std"]),pre["version"]); x=norm.transform(selected); xi,_=quantize_int8(x,float(q["input_scale"])); input_json={"format_version":"sparrowml_multilayer_input_v1","samples":[{"sample_id":e.sample_id,"expected_label":e.class_id,"input_int8":row.astype(int).tolist()} for e,row in zip(selected,xi,strict=True)]}; _json(package/"input.json",input_json); (package/"input_data.bin").write_bytes(_i8(xi))
    payload={"fc1_weights":_i8(ir["constants"]["fc1_weights"]),"fc1_bias":_i32(ir["constants"]["fc1_bias"]),"fc1_scales":_f32(ir["constants"]["fc1_scales"]),"fc2_weights":_i8(ir["constants"]["fc2_weights"]),"fc2_bias":_i32(ir["constants"]["fc2_bias"]),"fc2_scales":_f32(ir["constants"]["fc2_scales"])}; image=bytearray(memory["total_memory_bytes"])
    for r in memory["regions"]:
        if r["name"] in payload: image[r["byte_offset"]:r["byte_offset"]+r["byte_length"]]=payload[r["name"]]
    (package/"model_data.bin").write_bytes(image); traces=[]
    for item,row in zip(input_json["samples"],xi,strict=True): traces.append({**item,**infer(row,q)})
    _json(package/"intermediate_reference.json",{"format_version":"phase6_export_trace_v1","samples":traces}); _json(package/"expected_output.json",{"format_version":"phase6_expected_output_v1","samples":[{"sample_id":x["sample_id"],"fc2_acc_int32":x["fc2_acc_int32"],"output_logits":x["output_logits"],"predicted_class":x["predicted_class"]} for x in traces]}); _json(package/"program.json",{"format_version":"sparrowml_symbolic_program_v2","commands":[{"operation":"load input","executor":"vector hardware"},{"operation":"load fc1 weights/bias/scales","executor":"vector hardware"},{"operation":"execute 16 dense dot products","executor":"vector hardware"},{"operation":"apply ReLU","executor":"scalar runtime software"},{"operation":"requantize hidden activations","executor":"scalar runtime software"},{"operation":"load fc2 weights/bias/scales","executor":"vector hardware"},{"operation":"execute four dense dot products","executor":"vector hardware"},{"operation":"store final outputs","executor":"host future adapter"}]})
    files=["model_ir.json","memory_map.json","model_data.bin","input_data.bin","input.json","expected_output.json","intermediate_reference.json","program.json"]; manifest={"package_format_version":c["package_version"],"model_name":ir["model_name"],"ir_version":IR_VERSION,"target_name":target["name"],"memory_region_summary":memory["regions"],"file_hashes":{n:_sha(package/n) for n in files},"no_sparrowv_execution":True}; _json(package/"manifest.json",manifest); (package/"README.md").write_text("# Phase 6 multi-layer package\n\nDense two-layer data package. It contains no Sparrow-V execution result.\n",encoding="utf-8"); valid=validate_export(package); _json(package/"export_report.json",valid); return package
def validate_export(package: Path) -> dict[str, Any]:
    manifest=json.loads((package/"manifest.json").read_text()); ir=json.loads((package/"model_ir.json").read_text()); validate_ir(ir); memory=json.loads((package/"memory_map.json").read_text()); data=(package/"model_data.bin").read_bytes(); previous=0
    for r in memory["regions"]:
        if r["byte_offset"]<previous or r["byte_offset"]%r["alignment"]: raise ValueError("overlapping Phase 6 memory map")
        previous=r["byte_offset"]+r["byte_length"]
    if len(data)!=memory["total_memory_bytes"]: raise ValueError("model image size mismatch")
    region={r["name"]:r for r in memory["regions"]}
    def read(name:str)->bytes: r=region[name]; return data[r["byte_offset"]:r["byte_offset"]+r["byte_length"]]
    constants=ir["constants"]
    if read("fc1_weights")!=_i8(constants["fc1_weights"]) or read("fc2_weights")!=_i8(constants["fc2_weights"]) or read("fc1_bias")!=_i32(constants["fc1_bias"]) or read("fc2_bias")!=_i32(constants["fc2_bias"]): raise ValueError("decoded tensors differ from IR")
    q={"input_scale":ir["quantization"]["input_scale"],"hidden_scale":ir["quantization"]["hidden_scale"],"layers":{"fc1":{"weight_int8":constants["fc1_weights"],"bias_int32":constants["fc1_bias"],"weight_scales":constants["fc1_scales"]},"fc2":{"weight_int8":constants["fc2_weights"],"bias_int32":constants["fc2_bias"],"weight_scales":constants["fc2_scales"]}}}; inputs=json.loads((package/"input.json").read_text())["samples"]; expected=json.loads((package/"intermediate_reference.json").read_text())["samples"]
    decoded_inputs=np.frombuffer((package/"input_data.bin").read_bytes(),dtype=np.int8).reshape(len(inputs),16)
    if [row.astype(int).tolist() for row in decoded_inputs] != [item["input_int8"] for item in inputs]: raise ValueError("decoded input differs from input.json")
    for item,actual in zip(inputs,expected,strict=True):
        reproduced=infer(np.asarray(item["input_int8"],dtype=np.int8),q)
        if any(reproduced[k]!=actual[k] for k in ("fc1_acc_int32","hidden_int8","fc2_acc_int32","predicted_class")): raise ValueError("package intermediate reference mismatch")
    for name,digest in manifest["file_hashes"].items():
        if _sha(package/name)!=digest: raise ValueError("package hash mismatch")
    return {"format_version":"phase6_export_validation_v1","passed":True,"reference_equivalence":"exact","samples":len(inputs),"hidden_buffer_bytes":region["hidden"]["byte_length"]}
def run_baseline(config_path: str | Path | None = None) -> dict[str, Any]:
    c,root=_config(config_path); output=_paths(c,root)[1]; metrics=train(config_path); calibrate(config_path); quantize(config_path); evaluation=evaluate(config_path); package=export(config_path); first={n:_sha(package/n) for n in ("model_ir.json","memory_map.json","model_data.bin","input_data.bin","expected_output.json")}; package=export(config_path); second={n:_sha(package/n) for n in first}
    if first!=second: raise ValueError("Phase 6 export is not deterministic")
    validation=validate_export(package); _json(output/"determinism.json",{"repeated_export_hashes":first,"equal":True}); (output/"summary.md").write_text("# Phase 6 multi-layer INT8\n\nSynthetic-fixture FP32 and integer reference workflow completed; no Sparrow-V execution occurred.\n",encoding="utf-8"); return {"training":metrics,"evaluation":evaluation,"export":validation}
