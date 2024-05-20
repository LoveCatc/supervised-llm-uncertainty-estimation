from pathlib import Path

import datasets

MMLU_TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def prepare_triviaqa(save_dir):
    # we follow the setting in semantic_uncertainty
    ds = datasets.load_dataset("trivia_qa", "rc.nocontext")
    ds.save_to_disk(str(Path(save_dir) / "trivia_qa"))


def prepare_coqa(save_dir):
    ds = datasets.load_dataset("stanfordnlp/coqa")
    ds.save_to_disk(str(Path(save_dir) / "coqa"))


def prepare_mmlu(save_dir):
    for task in MMLU_TASKS:
        ds = datasets.load_dataset("lukaemon/mmlu", task)
        ds.save_to_disk(str(Path(save_dir) / "mmlu" / task))


def prepare_wmt(save_dir):
    ds = datasets.load_dataset("wmt14", "fr-en")
    ds.save_to_disk(str(Path(save_dir) / "wmt14"))
