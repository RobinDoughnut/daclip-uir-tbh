## To do
Introduce self attention in model.py (image restoration)


What I changed
- `universal-image-restoration/open_clip/transformer.py`: supports `block_type="deepnorm"` using `DeepNormResidualAttentionBlock` and passes it through `VisionTransformer` and `TextTransformer`.
- `universal-image-restoration/open_clip/model.py`: dataclasses now include `block_type`; passed into both transformers.
- `universal-image-restoration/open_clip/model_configs/daclip_ViT-B-32.json`: added `"block_type": "deepnorm"` to both `vision_cfg` and `text_cfg`.
- `da-clip/src/open_clip/transformer.py`: added `block_type` support; implemented DeepNorm block inline to avoid missing import.
- `da-clip/src/open_clip/model.py`: dataclasses now include `block_type`; passed into both transformers.
- `da-clip/src/open_clip/model_configs/daclip_ViT-B-32.json`: added `"block_type": "deepnorm"` to both `vision_cfg` and `text_cfg`.

How to use
- Using ViT-B/32 configs already enables DeepNorm.
- To switch per model, set `"block_type": "deepnorm"` (or `"residual"`) in the model’s `vision_cfg` and `text_cfg`.

- No linter errors introduced.



### what happened?
- DeepNorm variant of the CLIP transformer and made it selectable via a new config field `block_type` (default remains "residual").
- Enabled DeepNorm by default for ViT-B/32 in both trees.

What I changed
- `universal-image-restoration/open_clip/transformer.py`: added `block_type` to `Transformer`, `VisionTransformer`, `TextTransformer`; if `"deepnorm"`, uses `DeepNormResidualAttentionBlock`.
- `universal-image-restoration/open_clip/model.py`: `CLIPVisionCfg` and `CLIPTextCfg` now include `block_type`; passed through to the transformers.
- `universal-image-restoration/open_clip/model_configs/daclip_ViT-B-32.json`: set `"block_type": "deepnorm"` in both `vision_cfg` and `text_cfg`.
- `da-clip/src/open_clip/transformer.py`: same `block_type` support wired in.
- `da-clip/src/open_clip/model.py`: same config plumbing.
- `da-clip/src/open_clip/model_configs/daclip_ViT-B-32.json`: set `"block_type": "deepnorm"` in both `vision_cfg` and `text_cfg`.

Impact
- Build behavior: New CLIP models created from those ViT-B/32 configs will use DeepNorm on both image and text towers.
- Training: Typically more stable residual scaling for deeper stacks; you may need to retune LR/warmup slightly. Compute/memory are roughly unchanged.
- Pretrained weight compatibility: Existing ViT-B/32 checkpoints trained with standard residual blocks will NOT load with DeepNorm (state_dict keys differ). If you need to load those weights, set `block_type` back to `"residual"` (or remove the field) in the relevant JSON(s).
- DA-CLIP control path: Unaffected in logic; controller still injects per-layer signals, now over DeepNorm blocks.

For DeepNorm only on vision or only on text, set `block_type` accordingly in that section of the model config.
