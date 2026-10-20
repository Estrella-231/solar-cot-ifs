# Attempt1 failure: metadata serialization only

PBS 215241.tc6000, exit_status=1. No R or GHI training started. CPU pairing and GPU kernel witnesses passed. Source code retained at commit 833a561 and remote code_attempt1. Fix: explicitly serialize Python int/bool for supervised-row counts. New runtime/code directories used for attempt2.

```text
  File "/public/home/slfu/miniconda3/envs/swc/lib/python3.10/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
  File "/public/home/slfu/miniconda3/envs/swc/lib/python3.10/json/encoder.py", line 201, in encode
    chunks = list(chunks)
  File "/public/home/slfu/miniconda3/envs/swc/lib/python3.10/json/encoder.py", line 431, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/public/home/slfu/miniconda3/envs/swc/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/public/home/slfu/miniconda3/envs/swc/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/public/home/slfu/miniconda3/envs/swc/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/public/home/slfu/miniconda3/envs/swc/lib/python3.10/json/encoder.py", line 438, in _iterencode
    o = _default(o)
  File "/public/home/slfu/miniconda3/envs/swc/lib/python3.10/json/encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type int64 is not JSON serializable
```
