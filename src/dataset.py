import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence

from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

from src.utils.env_utils import DEFAULT_DATA_DIR, GPT_4O_CACHE_DIR
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelationSample(DataClassJsonMixin):
    subject: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} -> {self.object}"


@dataclass(frozen=False)
class InContextQuery(DataClassJsonMixin):
    """A query with a subject and object, and a context in which to embed it."""

    subject: str
    cf_description: str
    answer: str

    template: str = (
        "Assume an alternative universe where <subj> is in <loc>. In that universe, <subj> is located in the city of"
    )

    def set_template(self, template: str):
        self.template = template

    @property
    def query(self) -> str:
        return self.template.replace("<subj>", self.subject).replace(
            "<loc>", self.cf_description
        )

    def __str__(self) -> str:
        return f"{self.subject} -> {self.cf_description} | answer: {self.answer}"


@dataclass(frozen=True)
class RelationProperties(DataClassJsonMixin):
    """Some metadata about a relation."""

    relation_type: str
    domain_name: str
    range_name: str
    symmetric: bool
    # fn_type: str
    # disambiguating: bool


@dataclass(frozen=False)
class Relation(DataClassJsonMixin, Dataset):
    """An abstract mapping between subjects and objects.

    Attributes:
        name: The name of the relation, used as an ID.
        prompt_templates: Prompts representing the relation, where the subject is
            represented by {}.
        samples: A list of (subject, object) pairs satisfying the relation.
        properties: Relation metadata.
        _domain: Explicit list of all possible subjects. Accessed via the @property
            `domain`, which guesses the domain from the samples if not provided.
        _range: Equivalent to `_domain`, but for objects.
    """

    name: str
    prompt_templates: list[str]
    prompt_templates_zs: list[str]
    samples: list[RelationSample]
    properties: RelationProperties

    _prompt_template_idx: int = 0  # use the first prompt template by default
    _few_shot_samples: list[RelationSample] = field(default_factory=list)
    _few_shot_prefix: str | None = None
    _domain: list[str] | None = None
    _range: list[str] | None = None

    def __post_init__(self):
        if len(self._few_shot_samples) == 0:
            self.select_icl_examples(5)

        logger.info(f'initialized relation -> "{self.name}" with {len(self)} samples')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        query = self.prompt_templates[self._prompt_template_idx].format(
            self.samples[idx].subject
        )
        object = self.samples[idx].object
        full_query = self._few_shot_prefix + "\n" + query
        return (full_query, object)

    def select_icl_examples(self, num_icl):
        # Select few shot samples
        self._few_shot_samples = random.sample(
            self.samples + self._few_shot_samples, num_icl
        )
        self._few_shot_prefix = "\n".join(
            [
                self.prompt_templates[self._prompt_template_idx].format(demo.subject)
                + " "
                + demo.object
                for demo in self._few_shot_samples
            ]
        )
        self.samples = list(set(self.samples) - set(self._few_shot_samples))

    @property
    def domain(self) -> set[str]:
        if self._domain is not None:
            return set(self._domain)
        return {sample.subject for sample in self.samples}

    @property
    def range(self) -> set[str]:
        if self._range is not None:
            return set(self._range)
        return {sample.object for sample in self.samples}

    def without(self, sample: RelationSample) -> "Relation":
        """Return a copy of this relation without a given sample."""
        return self.set(samples=[s for s in self.samples if s != sample])

    def split(
        self, train_size: int, test_size: int | None = None
    ) -> tuple["Relation", "Relation"]:
        """Break into a train/test split."""
        if train_size > len(self.samples):
            raise ValueError(f"size must be <= {len(self.samples)}, got: {train_size}")
        if test_size is None:
            test_size = len(self.samples) - train_size

        # Shuffle once up front, because we're sometimes sorted, and if the relation
        # is 1:1, we'll always pick the same samples!
        samples = self.samples.copy()
        random.shuffle(samples)

        samples_by_object = defaultdict(list)
        for sample in samples:
            samples_by_object[sample.object].append(sample)

        for samples in samples_by_object.values():
            random.shuffle(samples)

        # List to store the result
        max_coverage_samples = []

        # As long as there are samples left
        while samples_by_object:
            # For each object
            for object in list(samples_by_object.keys()):
                # Add one sample to the result and remove it from the object's list
                max_coverage_samples.append(samples_by_object[object].pop(0))

                # If there are no more samples for this object, remove it from the dict
                if len(samples_by_object[object]) == 0:
                    del samples_by_object[object]

        train_samples = max_coverage_samples[:train_size]
        test_samples = max_coverage_samples[train_size : train_size + test_size]

        return (
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                prompt_templates_zs=self.prompt_templates_zs,
                properties=self.properties,
                samples=train_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                prompt_templates_zs=self.prompt_templates_zs,
                properties=self.properties,
                samples=test_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
        )

    def set(
        self,
        name: str | None = None,
        prompt_templates: Sequence[str] | None = None,
        prompt_templates_zs: Sequence[str] | None = None,
        properties: RelationProperties | None = None,
        samples: Sequence[RelationSample] | None = None,
        domain: Sequence[str] | None = None,
        range: Sequence[str] | None = None,
    ) -> "Relation":
        """Return a copy of this relation with any specified fields overwritten."""
        return Relation(
            name=name if name is not None else self.name,
            prompt_templates=(
                list(prompt_templates)
                if prompt_templates is not None
                else self.prompt_templates
            ),
            prompt_templates_zs=(
                list(prompt_templates_zs)
                if prompt_templates_zs is not None
                else self.prompt_templates_zs
            ),
            properties=properties if properties is not None else self.properties,
            samples=list(samples) if samples is not None else self.samples,
            _domain=list(domain) if domain is not None else self._domain,
            _range=list(range) if range is not None else self._range,
        )


class RelationDataset(Dataset[Relation], DataClassJsonMixin):
    """A torch dataset of relations."""

    def __init__(self, relations: list[Relation]):
        self.relations = relations

    def __len__(self) -> int:
        return len(self.relations)

    def __getitem__(self, index: int) -> Relation:
        return self.relations[index]

    def filter(
        self,
        relation_names: Sequence[str] | None = None,
        **properties: bool | Sequence[str],
    ) -> "RelationDataset":
        relations = list(self.relations)
        if relation_names is not None:
            logger.debug(f"filtering to only relations: {relation_names}")
            relations = [r for r in relations if r.name in set(relation_names)]

        for key, value in properties.items():
            if value is not None:
                if isinstance(value, bool):
                    logger.debug(f"filtering by property {key}={value}")
                    matches = lambda x: x == value
                else:
                    logger.debug(f"filtering by property {key} in {value}")
                    value_set = set(value)
                    matches = lambda x: (x in value_set)

                relations = [
                    r for r in relations if matches(getattr(r.properties, key))
                ]

        return RelationDataset(relations)

    def to_json(self):
        return [relation.to_dict() for relation in self.relations]

    def from_json(data: list[dict]):
        return RelationDataset(
            relations=[Relation.from_dict(relation) for relation in data]
        )


def resolve_relation_file_path(file_name: str) -> Path:
    """Resolve the path to a relation file."""
    relation_path = os.path.join(DEFAULT_DATA_DIR, "relation")
    relation_categories = os.listdir(relation_path)
    for category in relation_categories:
        category_dir = Path(relation_path) / category
        if not category_dir.is_dir():
            continue
        for file in os.listdir(category_dir):
            if file == file_name:
                return Path(relation_path) / category / file
    raise FileNotFoundError(f"could not find relation file for {file_name}")


def load_relation_dict(file: PathLike, absolute_path=False) -> dict:
    """Load dict for a single relation from a json file."""
    file = resolve_relation_file_path(file) if not absolute_path else Path(file)
    if file.suffix != ".json":
        raise ValueError(f"relation files must be json, got: {file}")
    with file.open("r") as handle:
        relation_dict = json.load(handle)
    for key in ("domain", "range"):
        if key in relation_dict:
            relation_dict[f"_{key}"] = relation_dict.pop(key)

    # Check that all keys are valid kwargs to Relation
    valid_keys = set(field.name for field in fields(Relation))
    for key in relation_dict.keys():
        if key not in valid_keys:
            raise ValueError(
                f"invalid key in relation file {file}: {key}. "
                f"valid keys are: {valid_keys}"
            )

    # Compute the type of relation function (injection, surjection, bijection, etc.)
    # relation_dict["properties"]["fn_type"] = get_relation_fn_type(relation_dict)

    return relation_dict


def load_relation(file: PathLike, absolute_path=False) -> Relation:
    """Load a single relation from a json file."""
    relation_dict = load_relation_dict(file, absolute_path=absolute_path)
    return Relation(
        name=relation_dict["name"],
        prompt_templates=relation_dict["prompt_templates"],
        prompt_templates_zs=relation_dict["prompt_templates_zs"],
        samples=[
            RelationSample.from_dict(sample) for sample in relation_dict["samples"]
        ],
        properties=RelationProperties.from_dict(relation_dict["properties"]),
    )


def load_relation_dataset(*paths: PathLike) -> RelationDataset:
    """Load relations from json files in a folder.

    Accepts one or more directories or files. If a file, should be JSON format, and will
    be read as one relation. If a directory, will recursively search for all JSON files.
    """
    if not paths:
        data_dir = os.path.join(DEFAULT_DATA_DIR, "relation")
        logger.debug(f"no paths provided, using default data dir: {data_dir}")
        paths = (data_dir,)

    # Load all relation files
    files = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            logger.debug(f"found relation file: {path}")
            files.append(path)
        else:
            logger.debug(f"{path} is directory, globbing for json files...")
            for file in sorted(path.glob("**/*.json")):
                logger.debug(f"found relation file: {file}")
                files.append(file)

    logger.debug(f"found {len(files)} relation files total, loading...")
    relation_dicts = [load_relation_dict(file, absolute_path=True) for file in files]

    # Mark all disambiguating relations
    domain_range_pairs: dict[tuple[str, str], int] = {}
    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        cur = domain_range_pairs.get((d, r), 0)
        domain_range_pairs[(d, r)] = cur + 1

    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        relation_dict["properties"]["disambiguating"] = domain_range_pairs[(d, r)] > 1

    # Create Relation objects
    relations = [Relation.from_dict(relation_dict) for relation_dict in relation_dicts]

    return RelationDataset(relations)


# Bridge Dataset ----------------------------------- #
@dataclass(frozen=False)
class BridgeSample(DataClassJsonMixin):
    bridge: str
    entity_pair: list[str]
    description: Optional[str] = None

    def __post_init__(self):
        assert (
            len(self.entity_pair) == 2
        ), f"entity_pair must have length 2, got {len(self.entity)} - {self.entity}"

    def __str__(self):
        return (
            self.description
            if self.description is not None
            else f"{self.bridge} is a common link between {self.entity[0]} and {self.entity[1]}."
        )


@dataclass(frozen=False)
class BridgeRelation(DataClassJsonMixin):
    name: str
    answer_template: str
    swappable: bool
    examples: list[BridgeSample] = field(default_factory=list)

    def __post_init__(self):
        assert "<bridge>" in self.answer_template
        assert "<entity1>" in self.answer_template
        assert "<entity2>" in self.answer_template
        for example in self.examples:
            example.description = (
                self.answer_template.replace("<bridge>", example.bridge)
                .replace("<entity1>", example.entity_pair[0])
                .replace("<entity2>", example.entity_pair[1])
            )
        logger.info(
            f"initialized bridge relation {self.name} with {len(self.examples)} examples"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass(frozen=False)
class BridgeDataset(DataClassJsonMixin):
    relations: list[BridgeRelation]
    examples: list[BridgeSample]
    icl_examples: list[BridgeSample]

    query_instruction: str = "Given two entities, find a common link between them."
    query_template: str = "What is a common link between <entity1> and <entity2>?"
    _prefix: Optional[str] = None

    def __init__(
        self,
        relations: list[BridgeRelation],
        examples: Optional[list[BridgeSample]] = None,
        icl_examples: Optional[list[BridgeSample]] = None,
        query_instruction: Optional[str] = None,
        query_template: Optional[str] = None,
        _prefix: Optional[str] = None,
    ):
        self.relations = relations

        if icl_examples is not None:
            self.icl_examples = icl_examples
        else:
            self.select_icl_examples(len(relations))

        if examples is not None:
            self.examples = examples
        else:
            self.examples = []
            for relation in relations:
                self.examples.extend(relation.examples)
            random.shuffle(self.examples)

        if query_instruction is not None:
            self.query_instruction = query_instruction

        if query_template is not None:
            self.query_template = query_template

        if _prefix is not None:
            self._prefix = _prefix

        logger.info(
            f"initialized bridge dataset with {len(self.relations)} relations and {len(self)} examples"
        )

    # @classmethod
    # def from_dict(dct: dict) -> "BridgeDataset":
    #     relations = [BridgeRelation.from_dict(r) for r in dct["relations"]]
    #     examples = [BridgeSample.from_dict(e) for e in dct["examples"]]
    #     icl_examples = [BridgeSample.from_dict(e) for e in dct["icl_examples"]]
    #     query_instruction = dct["query_instruction"]
    #     query_template = dct["query_template"]
    #     _prefix = dct["_prefix"]

    #     return BridgeDataset()

    #     logger.info(
    #         f"loaded bridge dataset with {len(self.relations)} relations and {len(self)} examples"
    #     )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]
        return (
            self.prefix
            + self.query_template.replace("<entity1>", sample.entity_pair[0]).replace(
                "<entity2>", sample.entity_pair[1]
            )
            + "\n"
            + "A:",
            sample.description,
        )

    def select_icl_examples(self, n: Optional[int] = None):
        if n is None:
            n = len(self.relations)
        elif n < len(self.relations):
            logger.warning(
                f"selecting {n} examples from {len(self.relations)} relations"
            )

        self.icl_examples = []
        r_idx = 0
        while n > 0:
            relation = self.relations[r_idx]
            sample_idx = random.randint(0, len(relation) - 1)
            self.icl_examples.append(relation[sample_idx])
            relation.examples.pop(sample_idx)

            r_idx = (r_idx + 1) % len(self.relations)
            n -= 1

        self._prefix = None

    @property
    def prefix(self):
        if self._prefix is not None:
            return self._prefix
        ins = self.query_instruction + "\n#\n"
        for sample in self.icl_examples:
            ins += (
                self.query_template.replace("<entity1>", sample.entity_pair[0]).replace(
                    "<entity2>", sample.entity_pair[1]
                )
                + "\n"
            )
            ins += "A: " + str(sample) + "\n#\n"
        return ins


def load_bridge_relation(file_name: str) -> BridgeRelation:
    with open(file_name, "r") as f:
        data = json.load(f)
    return BridgeRelation.from_dict(data)


def load_bridge_relations() -> list[BridgeRelation]:
    bridge_data_dir = os.path.join(DEFAULT_DATA_DIR, "bridge_dataset", "cleaned")
    relations = []
    for file_name in os.listdir(bridge_data_dir):
        if file_name.endswith(".json"):
            relations.append(
                load_bridge_relation(os.path.join(bridge_data_dir, file_name))
            )
    return relations


def load_bridge_dataset() -> BridgeDataset:
    relations = load_bridge_relations()
    return BridgeDataset(relations)


# Bridge Dataset ----------------------------------- #
