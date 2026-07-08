"""
nnUNet v1 Dataset Preparation Script -- Mouse Brain Segmentation
=================================================================
Configuration restored from debug.json:
  - 51 label values (50 brain regions + background 0)
  - 5-modality channel order: QSMs(0) / T1(1) / T2(2) / T2s(3) / iMag(4)
  - Single-modality Tasks each use the corresponding 1 channel

Expected directory layout (set BASE_DIR to your local path):
  RAW_DATA_BASE/
    Task020_Mouse5mod/
      imagesTr/   <- training images, named case_XXXX_0000.nii.gz ... _0004.nii.gz
      labelsTr/   <- training labels, named case_XXXX.nii.gz
      imagesTs/   <- test images (may be empty)

Usage:
  python 01_prepare_dataset.py
"""

import os
import json

# ── Set to your local data path ──────────────────────────────────────────────
BASE_DIR = "/path/to/data"
RAW_DATA_BASE = os.path.join(BASE_DIR, "nnUNet_raw_data_base", "nnUNet_raw_data")
# ─────────────────────────────────────────────────────────────────────────────

# 50 mouse brain region labels (index 1-50), plus background 0
LABELS = {
    "0": "background",
    "1": "region_01", "2": "region_02", "3": "region_03", "4": "region_04",
    "5": "region_05", "6": "region_06", "7": "region_07", "8": "region_08",
    "9": "region_09", "10": "region_10", "11": "region_11", "12": "region_12",
    "13": "region_13", "14": "region_14", "15": "region_15", "16": "region_16",
    "17": "region_17", "18": "region_18", "19": "region_19", "20": "region_20",
    "21": "region_21", "22": "region_22", "23": "region_23", "24": "region_24",
    "25": "region_25", "26": "region_26", "27": "region_27", "28": "region_28",
    "29": "region_29", "30": "region_30", "31": "region_31", "32": "region_32",
    "33": "region_33", "34": "region_34", "35": "region_35", "36": "region_36",
    "37": "region_37", "38": "region_38", "39": "region_39", "40": "region_40",
    "41": "region_41", "42": "region_42", "43": "region_43", "44": "region_44",
    "45": "region_45", "46": "region_46", "47": "region_47", "48": "region_48",
    "49": "region_49", "50": "region_50",
}

# ── Task definitions ──────────────────────────────────────────────────────────
# Each entry: task_name, modality_dict, training/validation case lists
# Replace region_XX strings in LABELS with actual anatomical names if available
TASKS = [
    {
        "name": "Task020_Mouse5mod",
        "description": "Mouse brain segmentation — 5 modalities (QSMs+T1+T2+T2s+iMag)",
        "tensorImageSize": "3D",
        "modality": {
            "0": "QSMs",
            "1": "T1",
            "2": "T2",
            "3": "T2s",
            "4": "iMag",
        },
        # Training case IDs (restored from training_log, 37 cases total)
        "training_cases": [
            "w20190929_092616MGEs80001a001", "w20190929_104242MGEs50001a001",
            "w20190929_125158MGEs60001a001", "w20190929_144136MGEs90001a001",
            "w20190929_163915MGEs60001a001", "w20190929_174817MGEs60001a001",
            "w20190929_184526MGEs60001a001", "w20190929_203552MGEs80001a001",
            "w20191006_103531MGEs80001a001", "w20191006_113944MGEs60001a001",
            "w20191006_134349MGEs40001a001", "w20191006_144359MGEs40001a001",
            "w20191006_154454MGEs40001a001", "w20191006_164542MGEs40001a001",
            "w20191006_174644MGEs40001a001", "w20191006_185221MGEs40001a001",
            "w20191006_195335MGEs50001a001", "w20191009_120043MGEs30001a001",
            "w20191009_120043MGEs70001a001", "w20191009_131839MGEs60001a001",
            "w20191009_142515MGEs40001a001", "w20191009_152924MGEs40001a001",
            "w20191009_180543MGEs60001a001", "w20191009_193654MGEs40001a001",
            "w20191027_171956MGEs40001a001", "w20191111_111733MGEs30001a001",
            "w20191111_121952MGEs30001a001", "w20191111_132007MGEs20001a001",
            "wcon_20191111_093351MGEs50001a001", "wcon_20191111_101345MGEs40001a001",
            "wlh_20191006_210301MGEs30001a001", "wlh_20191006_213508MGEs40001a001",
            "wlh_20191006_224410MGEs40001a001", "wlh_20191009_163354MGEs20001a001",
            "wlh_20191009_170503MGEs40001a001", "wlh_20191009_173510MGEs50001a001",
            "wlh_20191009_190643MGEs40001a001",
        ],
        # Validation case IDs (10 cases, from training_log)
        "validation_cases": [
            "w20190929_115000MGEs50001a001", "w20190929_193910MGEs60001a001",
            "w20191006_124207MGEs50001a001", "w20191009_093915MGEs50001a001",
            "w20191009_105237MGEs40001a001", "w20191027_181940MGEs40001a001",
            "w20191027_192010MGEs40001a001", "w20191111_115016MGEs50001a001",
            "w20191111_124732MGEs30001a001", "wcon_20191111_104745MGEs30001a001",
        ],
    },
    {
        "name": "Task021_Mouse_T2",
        "description": "Mouse brain segmentation — single modality T2",
        "tensorImageSize": "3D",
        "modality": {"0": "T2"},
        "training_cases": None,   # same cases as Task020, images contain T2 only
        "validation_cases": None,
    },
    {
        "name": "Task022_Mouse_T2s",
        "description": "Mouse brain segmentation — single modality T2s",
        "tensorImageSize": "3D",
        "modality": {"0": "T2s"},
        "training_cases": None,
        "validation_cases": None,
    },
    {
        "name": "Task023_Mouse_T1",
        "description": "Mouse brain segmentation — single modality T1",
        "tensorImageSize": "3D",
        "modality": {"0": "T1"},
        "training_cases": None,
        "validation_cases": None,
    },
    {
        "name": "Task024_Mouse_iMag",
        "description": "Mouse brain segmentation — single modality iMag",
        "tensorImageSize": "3D",
        "modality": {"0": "iMag"},
        "training_cases": None,
        "validation_cases": None,
    },
    {
        "name": "Task025_Mouse_QSM",
        "description": "Mouse brain segmentation — single modality QSMs",
        "tensorImageSize": "3D",
        "modality": {"0": "QSMs"},
        "training_cases": None,
        "validation_cases": None,
    },
    {
        "name": "Task026_ContinueLearn_Fold0",
        "description": "Mouse brain segmentation — continual learning from Task020 fold0",
        "tensorImageSize": "3D",
        "modality": {
            "0": "QSMs",
            "1": "T1",
            "2": "T2",
            "3": "T2s",
            "4": "iMag",
        },
        "training_cases": None,   # new dataset for continual learning; replace with actual case IDs
        "validation_cases": None,
    },
]


def get_training_cases_from_dir(task_dir: str):
    """Infer case list from imagesTr directory (used when training_cases=None)."""
    images_dir = os.path.join(task_dir, "imagesTr")
    if not os.path.isdir(images_dir):
        return []
    cases = set()
    for fname in os.listdir(images_dir):
        if fname.endswith(".nii.gz"):
            # strip modality suffix _XXXX.nii.gz
            case_id = "_".join(fname.replace(".nii.gz", "").split("_")[:-1])
            cases.add(case_id)
    return sorted(cases)


def make_dataset_json(task: dict):
    task_name = task["name"]
    task_dir = os.path.join(RAW_DATA_BASE, task_name)
    os.makedirs(os.path.join(task_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(task_dir, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(task_dir, "imagesTs"), exist_ok=True)

    training_cases = task["training_cases"]
    validation_cases = task["validation_cases"] or []

    if training_cases is None:
        # reuse the same case split as Task020
        training_cases = TASKS[0]["training_cases"]
        validation_cases = TASKS[0]["validation_cases"]

    all_cases = list(training_cases) + list(validation_cases)
    num_modalities = len(task["modality"])

    training_entries = []
    for case in all_cases:
        if num_modalities == 1:
            image_path = f"./imagesTr/{case}_0000.nii.gz"
        else:
            # multi-modal: list all modality channels
            image_path = [f"./imagesTr/{case}_{str(m).zfill(4)}.nii.gz"
                          for m in range(num_modalities)]
        training_entries.append({
            "image": image_path,
            "label": f"./labelsTr/{case}.nii.gz",
        })

    dataset_json = {
        "name": task_name,
        "description": task["description"],
        "tensorImageSize": task["tensorImageSize"],
        "reference": "",
        "licence": "",
        "release": "1.0",
        "modality": task["modality"],
        "labels": LABELS,
        "numTraining": len(all_cases),
        "numTest": 0,
        "training": training_entries,
        "test": [],
    }

    out_path = os.path.join(task_dir, "dataset.json")
    with open(out_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"[OK] {task_name}: dataset.json written → {out_path}")
    print(f"     {len(all_cases)} cases ({len(training_cases)} train / {len(validation_cases)} val), "
          f"{num_modalities} modality")


if __name__ == "__main__":
    print(f"RAW_DATA_BASE = {RAW_DATA_BASE}\n")
    for task in TASKS:
        make_dataset_json(task)
    print("\nDone. Run 02_train.sh next for preprocessing and training.")
