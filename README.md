# Subject Motion 6DoF

Subject Motion 6DoF trains a causal video model that turns a streaming video
into one rigid-body 6DoF motion trajectory for the intended subject.

The intended subject may be one person among multiple people in frame. The
model does not require an explicit segmentation mask or an explicit person-ID
track. Instead, it uses human pose estimation features as a prior and learns,
from supervised video/label pairs, which subject-motion signal should be
predicted over time.

Conceptually, the selected person is abstracted into a single moving rigid
column:

```text
x, y, z, roll, pitch, yaw
```

The output is a `.json` motion file containing those six normalized channels.

## Task

Given a video stream, predict the normalized 6DoF motion of the implicit subject
at each frame.

The model is designed around these assumptions:

- The input is causal: predictions at frame `t` use frames up to `t`, not future
  frames.
- The subject can be implicit: the training label defines the desired subject
  and motion target, even when the frame contains multiple people.
- Human pose features act as a prior: pose-token evidence helps the model focus
  on person-like motion without requiring a separate segmentation or tracking
  stage.
- The output is a compact rigid-body abstraction, not a full mesh, skeleton, or
  per-joint reconstruction.

## Label Format

Training data is expected as same-stem video and JSON annotation pairs:

```text
example.mp4
example.json
```

The JSON stores sparse action points for each 6DoF channel:

```json
{
  "channels": {
    "x": {"actions": [{"at": 0, "pos": 50}]},
    "y": {"actions": [{"at": 0, "pos": 50}]},
    "z": {"actions": [{"at": 0, "pos": 50}]},
    "roll": {"actions": [{"at": 0, "pos": 50}]},
    "pitch": {"actions": [{"at": 0, "pos": 50}]},
    "yaw": {"actions": [{"at": 0, "pos": 50}]}
  }
}
```

Channel meanings:

```text
channels.x.actions                 -> subject rigid-body translation x
channels.y.actions                 -> subject rigid-body translation y
channels.z.actions                 -> subject rigid-body translation z
channels.roll.actions              -> subject rigid-body rotation roll
channels.pitch.actions             -> subject rigid-body rotation pitch
channels.yaw.actions               -> subject rigid-body rotation yaw
```

Each action point uses:

```json
{"at": 1234, "pos": 50}
```

`at` is milliseconds. `pos` is a normalized supervised value clamped to
`0..100` by the default loader.

Sparse points are converted into dense per-frame targets with linear
interpolation. A channel with at least one action point is supervised on its
dense trajectory. A channel with no action points is ignored by the loss.

## Model

The main model is `SubjectMotion6DoFModel`.

Frames are resized with preserved aspect ratio, center padded to
`MODEL.IMAGE_SIZE`, normalized, and passed through:

```text
frames
-> DINOv3 backbone
-> SAM-3D Body PromptableDecoder
-> primary body pose token: tokens[:, 0]
-> causal token-buffered temporal head
-> six subject rigid-body 6DoF channels in 0..100
```

The visual stream uses SAM-3D Body pose features as the pose prior. The primary
body pose token has dimension `1024` and is treated as the per-frame visual
summary for downstream temporal prediction.

The temporal head has two parts:

- A causal visual temporal module over the most recent `DATA.MAX_LEN` pose
  tokens.
- An action-only autoregressive head that predicts each channel from the current
  visual context plus that channel's previous predicted value/state.

During streaming training and generation, the model carries:

```text
pose_token_buffer
prev_actions
action_state
```

`pose_token_buffer` stores visual history for subject consistency across time.
`prev_actions` and `action_state` carry the autoregressive motion state.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run scripts from the repository root. The scripts add the repository root to
`sys.path`, so an editable package install is not required.

The DINOv3 backbone is constructed through `torch.hub`. Make sure the required
DINOv3 code and SAM-3D weights are available before training or generation.

## Configuration

Default config:

```text
configs/subject_motion_6dof.yaml
```

Key defaults:

```yaml
MODEL:
  IMAGE_SIZE: [512, 512]
  DECODER:
    DIM: 1024
    DEPTH: 6
    HEADS: 8
    MLP_DIM: 1024
    DIM_HEAD: 64
    ENABLE_TWOWAY: false
    REPEAT_PE: true
    DO_HAND_DETECT_TOKENS: false

TEMPORAL_HEAD:
  INPUT_DIM: 1024
  HIDDEN_DIM: 512
  OUTPUT_DIM: 6

DATA:
  MAX_LEN: 240
  CHUNK_LEN: 128
  REQUIRE_CFR: true

TRAINING:
  FREEZE_BACKBONE: true
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 8
```

`DATA.MAX_LEN` controls how much visual pose-token history the causal temporal
module can see. `DATA.CHUNK_LEN` controls how many new frames are encoded per
training sample.

## Initialize Visual Weights

Create a model checkpoint initialized from a SAM-3D Body checkpoint:

```bash
python scripts/init_from_sam3d.py \
  --config configs/subject_motion_6dof.yaml \
  --source_ckpt checkpoints/sam3d_body.ckpt \
  --output_ckpt outputs/pretrained_subject_motion_6dof.pt
```

The initializer copies compatible visual backbone, decoder, prompt, token, and
ray-conditioning weights into the subject-motion model. The temporal motion
head remains task-specific and is trained on the supervised video/JSON pairs.

## Prepare CFR Videos

Training and generation are fail-closed for video timing by default. With
`DATA.REQUIRE_CFR: true`, videos must have verifiable constant frame rate.

Batch-convert a folder of paired videos and `.json` files:

```bash
python scripts/prepare_cfr_dataset.py \
  --input_root data/raw \
  --output_root data/prepared \
  --fps 30 \
  --encoder h264_nvenc
```

The script preserves relative subdirectories, converts videos to square
aspect-preserving padded MP4 files, and copies same-stem `.json` files next to
the converted videos.

Use `--overwrite` to replace existing outputs. Use `--allow_missing_json` when
preparing unlabeled videos for later generation or inspection.

## Train

```bash
python scripts/train.py \
  --config configs/subject_motion_6dof.yaml \
  --data_root data/prepared \
  --pretrained_ckpt outputs/pretrained_subject_motion_6dof.pt \
  --output_dir outputs/run1
```

Training is epoch based. Chunks are read in timestamp order with
`shuffle=False`, allowing the trainer to carry streaming state from one chunk
to the next for the same video.

Gradient flow is truncated at chunk boundaries because cached streaming state is
detached between chunks. Forward-time conditioning still continues across
sequential chunks.

Resume the latest complete checkpoint in an output directory by running the same
command again. Use `--resume` only when selecting a specific checkpoint:

```bash
python scripts/train.py \
  --config configs/subject_motion_6dof.yaml \
  --data_root data/prepared \
  --resume outputs/run1/checkpoint-2000 \
  --output_dir outputs/run1
```

Optional validation:

```bash
python scripts/train.py \
  --config configs/subject_motion_6dof.yaml \
  --data_root data/prepared_train \
  --val_data_root data/prepared_val \
  --pretrained_ckpt outputs/pretrained_subject_motion_6dof.pt \
  --output_dir outputs/run1
```

Validation only runs when `TRAINING.VALIDATE_EVERY` is greater than zero.

Optional Weights & Biases logging:

```bash
python scripts/train.py \
  --config configs/subject_motion_6dof.yaml \
  --data_root data/prepared \
  --pretrained_ckpt outputs/pretrained_subject_motion_6dof.pt \
  --output_dir outputs/run1 \
  --wandb \
  --wandb_project subject-motion-6dof
```

## Generate

```bash
python scripts/generate_motion_json.py \
  --config configs/subject_motion_6dof.yaml \
  --ckpt outputs/run1/checkpoint-50000/weights.pt \
  --video input.mp4 \
  --output output.json \
  --encode_batch_size 128
```

Generation is fully autoregressive. Frames are encoded in batches for
throughput, but the action head advances one frame at a time while carrying
`pose_token_buffer`, `prev_actions`, and `action_state`.

The output JSON uses the same six-channel convention as the training labels.
