import json
from pathlib import Path
import pytest
from uniphysics.data import DEFAULT_DATASET, load_dataset, dataset_summary
from uniphysics.extract import extract_answer, extract_judge
from uniphysics.rules import rule_decision
from uniphysics.metrics import calculate_metrics
from uniphysics.pipeline import run_directory, run_lock
from uniphysics.api import APIConfig


def test_packaged_datasets():
    clean = load_dataset(DEFAULT_DATASET)
    removed = load_dataset(DEFAULT_DATASET.with_name('UniPhysics_deleted.jsonl'))
    assert len(clean) == 5352
    assert len(removed) == 3751
    assert dataset_summary(removed)['empty_answers'] == 9
    assert len(load_dataset(DEFAULT_DATASET, 'MMLU_pro')) == 1035
    assert len(load_dataset(DEFAULT_DATASET, 'GPQA')) == 178
    assert len(load_dataset(DEFAULT_DATASET, test_mode='mini')) == 2514
    assert len({r['id'] for r in removed}) == len(removed)


@pytest.mark.parametrize('kind,candidate,gold,expected', [
    ('MCQ', 'B', 'B. 42 joules', True),
    ('MCQ', 'B', 'B', True),
    ('MCQ', 'J', 'J. answer', True),
    ('MCQ', 'A,B', 'A. 12', False),
    ('MCQ', '108', 'C. 108', True),
    ('MCQ(multiple)', 'J and A', 'A,J', True),
    ('MCQ(multiple)', 'A', 'A,D', False),
    ('Boolean', 'No, Yes', 'False; True', True),
    ('Boolean', 'Yes', 'No', False),
])
def test_rules(kind, candidate, gold, expected):
    assert rule_decision(kind, candidate, gold) is expected


def test_extraction():
    assert extract_answer(r'First \boxed{A}; final \boxed{\frac{a}{b}}')[0] == r'\frac{a}{b}'
    assert extract_answer('Final answer: 3.14159')[0] == '3.14159'
    assert extract_judge('<judge>False</judge>') == 'false'
    assert extract_judge('<judge>unknown</judge>') is None


def test_metrics_expose_failures_and_macro_weighting():
    records = [{'id': 'GPQA1', 'judge': True}, {'id': 'GPQA2', 'judge': False},
               {'id': 'GPQA3', 'judge': None, 'judge_status': 'api_error'},
               {'id': 'HiPhO1', 'judge': True}]
    metrics = calculate_metrics(records)
    assert metrics['micro_accuracy'] == pytest.approx(2/3)
    assert metrics['macro_accuracy'] == .75
    assert metrics['overall']['coverage'] == .75
    assert metrics['overall']['accuracy_over_all'] == .5
    assert metrics['overall']['api_errors'] == 1


def test_project_path_and_lock(tmp_path):
    with pytest.raises(ValueError):
        run_directory(tmp_path, '../escape')
    directory = run_directory(tmp_path, 'demo')
    with run_lock(directory):
        with pytest.raises(ValueError):
            with run_lock(directory):
                pass
    assert not (directory / '.run.lock').exists()


def test_invalid_configuration():
    with pytest.raises(ValueError):
        APIConfig('model', max_workers=0)
    from uniphysics.local import LocalConfig
    with pytest.raises(ValueError):
        LocalConfig('model', batch_size=0)
    with pytest.raises(ValueError):
        LocalConfig('model', max_model_len=2048, max_tokens=2048)


def test_schema_and_duplicate_errors(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text('{bad json}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='invalid JSON'):
        load_dataset(source)
    row = {'id': 'GPQA1', 'problem': 'q', 'answer': 'a', 'answer_type': 'MCQ'}
    source.write_text((json.dumps(row) + '\n') * 2, encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        load_dataset(source)
