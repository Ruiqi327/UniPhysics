import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from uniphysics.api import APIClient, APIConfig
from uniphysics.cli import main
from uniphysics.io import read_jsonl, write_jsonl
from uniphysics.pipeline import judge_stage


@pytest.fixture
def endpoint():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            model = body['model']
            prompt = body['messages'][0]['content']
            status = 200
            if model == 'unauthorized':
                status = 401
                response = {'error': {'message': 'server echoed private credential', 'type': 'auth_error'}}
            else:
                if model == 'generator':
                    content = r'\boxed{B}' if 'closed question' in prompt else r'\boxed{42}'
                elif model == 'broken':
                    content = 'no valid judge tag'
                else:
                    content = '<judge>False</judge>' if model == 'judge-two' and 'disagreement' in prompt else '<judge>True</judge>'
                response = {'id': 'test', 'object': 'chat.completion', 'created': 1,
                            'model': model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}]}
            data = json.dumps(response).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1', requests
    server.shutdown()
    server.server_close()
    thread.join()


def test_end_to_end_http_resume_and_judge_cache(tmp_path, monkeypatch, endpoint):
    url, requests = endpoint
    monkeypatch.setenv('UNIPHYSICS_API_KEY', 'test-only')
    rows = [
        {'id': 'GPQA1', 'problem': 'closed question', 'answer': 'B. 42', 'answer_type': 'MCQ'},
        {'id': 'GPQA2', 'problem': 'agreement', 'answer': '42', 'answer_type': 'Open-Ended'},
        {'id': 'GPQA3', 'problem': 'disagreement', 'answer': '42', 'answer_type': 'Open-Ended'},
        {'id': 'GPQA4', 'problem': 'missing reference', 'answer': '', 'answer_type': 'Open-Ended'},
    ]
    source = tmp_path / 'input.jsonl'
    write_jsonl(source, rows)
    common = ['--project_name', 'demo', '--output-dir', str(tmp_path), '--base-url', url]
    generation = ['api', *common, '--model', 'generator', '--dataset', str(source),
                  '--temperature', '0.1', '--top_p', '0.95']
    main(generation)
    assert len(requests) == 4
    assert all(r['temperature'] == .1 and r['top_p'] == .95 for r in requests)
    assert [r['id'] for r in read_jsonl(tmp_path / 'demo/response.jsonl')] == [r['id'] for r in rows]
    main([*generation, '--resume'])
    assert len(requests) == 4
    with pytest.raises(SystemExit) as caught:
        main(generation)
    assert caught.value.code == 2
    judge = ['judge', *common, '--model_1', 'judge-one', '--model_2', 'judge-two', '--model_3', 'arbiter']
    main(judge)
    assert len(requests) == 9  # 2 + 2 open-ended judgments and one arbitration
    result = read_jsonl(tmp_path / 'demo/judge.jsonl')
    assert [r['judge'] for r in result] == [True, True, True, None]
    assert result[-1]['judge_status'] == 'missing_gold'
    assert all('temperature' not in r and 'top_p' not in r for r in requests[4:])
    main([*judge, '--resume'])
    assert len(requests) == 9
    # A changed answer with the same ID must invalidate that judge cache.
    responses = read_jsonl(tmp_path / 'demo/response.jsonl')
    responses[1]['extracted_answer'] = '43'
    write_jsonl(tmp_path / 'demo/response.jsonl', responses)
    main([*judge, '--resume'])
    assert len(requests) == 11
    assert not any('test-only' in p.read_text() for p in (tmp_path / 'demo').glob('*.json*'))
    # Generation refuses a mismatched dataset on resume.
    rows[0]['problem'] += ' changed'
    write_jsonl(source, rows)
    with pytest.raises(SystemExit):
        main([*generation, '--resume'])


def test_api_retry_and_safe_failure(endpoint):
    url, requests = endpoint
    client = APIClient('test-only', url, APIConfig('unauthorized', max_retries=4, retry_wait=0))
    result = client.request('hello')
    client.close()
    assert not result.succeeded and result.attempts == 1
    assert 'credential' not in str(result.errors)
    client = APIClient('test-only', url, APIConfig('broken', mode='judge', max_retries=2, retry_wait=0))
    result = client.request('hello')
    client.close()
    assert not result.succeeded and result.attempts == 3


def test_partial_judge_cache_only_retries_failed_rows(tmp_path, endpoint):
    url, requests = endpoint
    records = [{'id': 'GPQA1', 'problem': 'p', 'answer': 'a', 'answer_type': 'Open-Ended'},
               {'id': 'GPQA2', 'problem': 'p', 'answer': 'a', 'answer_type': 'Open-Ended'}]
    config = APIConfig('judge-one')
    path = tmp_path / 'stage.jsonl'
    judge_stage(records, [0, 1], path, config, 'test-only', url, False)
    rows = read_jsonl(path)
    rows[1]['response'] = 'API_ERROR'
    write_jsonl(path, rows)
    judge_stage(records, [0, 1], path, config, 'test-only', url, True)
    assert len(requests) == 3


def test_rule_only_judging_requires_no_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv('UNIPHYSICS_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    write_jsonl(tmp_path / 'rules/response.jsonl', [
        {'id': 'GPQA1', 'problem': 'p', 'answer': 'B', 'answer_type': 'MCQ', 'response': r'\boxed{B}'},
        {'id': 'GPQA2', 'problem': 'p', 'answer': 'B', 'answer_type': 'MCQ', 'response': 'API_ERROR'},
    ])
    with pytest.raises(SystemExit) as caught:
        main(['judge', '--project_name', 'rules', '--output-dir', str(tmp_path)])
    assert caught.value.code == 1
    rows = read_jsonl(tmp_path / 'rules/judge.jsonl')
    assert rows[0]['judge'] is True
    assert rows[1]['judge'] is None and rows[1]['judge_status'] == 'generation_error'


def test_local_generation_contract_without_gpu(tmp_path, monkeypatch):
    from uniphysics import local
    monkeypatch.setattr(local, 'call_local_iter', lambda prompts, config: iter([(0, r'\boxed{B}')]))
    source = tmp_path / 'input.jsonl'
    write_jsonl(source, [{'id': 'GPQA1', 'problem': 'p', 'answer': 'B', 'answer_type': 'MCQ'}])
    main(['local', '--project_name', 'local', '--output-dir', str(tmp_path), '--dataset', str(source),
          '--model-path', 'example/model'])
    assert read_jsonl(tmp_path / 'local/response.jsonl')[0]['extracted_answer'] == 'B'
