"""Record real forward-hook shapes for the current 4-second LTRR configuration.

No dataset/checkpoint is loaded and no training is performed. The zero waveform
is only a shape probe; it is not an example separation result.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import yaml
    from loguru import logger
    logger.remove()
    from models.SepReformer_LTRR_VnSpeechMix_16K.model import Model

    package = ROOT / "models/SepReformer_LTRR_VnSpeechMix_16K"
    config = yaml.safe_load((package / "configs.yaml").read_text(encoding="utf-8"))["config"]
    torch.set_num_threads(2)
    model = Model(**config["model"]).eval()
    records = {}

    def shape(value):
        if isinstance(value, torch.Tensor):
            return list(value.shape)
        if isinstance(value, (tuple, list)):
            return [shape(item) for item in value]
        return str(value)

    names = ["audio_encoder", "feature_projector", "separator.pos_emb",
             "separator.bottleneck_G", "separator.spk_split_block", "separator",
             "ltrr", "ltrr.bottleneck", "ltrr.gcu", "ltrr.gcu.conv_u",
             "ltrr.gcu.conv_v", "ltrr.gcu.memory", "ltrr.gcu.memory.ffn_in",
             "ltrr.gcu.memory.out_proj", "ltrr.output_layer", "out_layer", "audio_decoder"]
    for index in range(4):
        names += [f"separator.enc_stages.{index}", f"separator.simple_fusion.{index}",
                  f"separator.dec_stages.{index}", f"ltrr.gcu.memory.proj_layers.{index}",
                  f"ltrr.gcu.memory.memory_layers.{index}", f"out_layer_bn.{index}",
                  f"decoder_bn.{index}"]
    handles = []
    for name in names:
        def hook(module, inputs, output, key=name):
            records.setdefault(key, []).append({"inputs": shape(inputs), "output": shape(output)})
        handles.append(model.get_submodule(name).register_forward_hook(hook))
    batch = config["dataloader"]["batch_size"]
    samples = config["dataset"]["max_len"]
    with torch.inference_mode():
        audio, auxiliary = model(torch.zeros(batch, samples), return_aux=True)
    for handle in handles:
        handle.remove()
    assert (batch, samples) == (2, 64000), "Update figure layout if the reference crop changes."
    assert records["audio_encoder"][0]["output"] == [2, 256, 7997]
    assert [v["output"][-1] for v in records["separator.spk_split_block"]] == [8000, 4000, 2000, 1000, 500]
    assert records["ltrr"][0]["inputs"] == [[4, 128, 8000]]
    assert records["ltrr"][0]["output"] == [4, 128, 8000]
    assert records["out_layer"][0]["output"] == [2, 2, 256, 7997]
    assert records["separator"][0]["output"][1] == [
        [4, 128, 500], [4, 128, 1000], [4, 128, 2000], [4, 128, 4000]]
    assert shape(audio) == [[2, 64000], [2, 64000]]
    files = [package / "configs.yaml", package / "model.py", package / "modules/module.py",
             package / "modules/network.py", package / "modules/ltrr.py"]
    libri = ROOT / "models/SepReformer_LTRR_Libri2Mix_16K"
    architecture_equal = all((package / name).read_bytes() == (libri / name).read_bytes()
                             for name in ["model.py", "modules/module.py", "modules/network.py", "modules/ltrr.py"])
    report = {
        "purpose": "Untrained CPU eval-mode shape probe, not a separation-quality measurement",
        "config": config["model"], "sampling_rate": config["dataset"]["sampling_rate"],
        "batch": batch, "samples": samples, "hooks": records,
        "audio_output": shape(audio), "auxiliary_audio_output": shape(auxiliary),
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "ltrr": sum(p.numel() for p in model.ltrr.parameters())},
        "alpha_initial": float(model.ltrr.alpha.detach()),
        "same_architecture_as_libri2mix_package": architecture_equal,
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in files},
    }
    output = ROOT / "figures/ltrr_shape_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified shapes; report: {output}")
    print("Parameters:", report["parameters"])
    print("Same architecture as Libri2Mix:", architecture_equal)


if __name__ == "__main__":
    main()
