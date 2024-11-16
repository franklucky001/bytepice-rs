# bytepiece algorithm of rust implement
> 根据苏剑林大神的实现 [bytepiece](https://github.com/bojone/bytepiece), 使用rust的重构版本, 包含`bytepiece_rs`lib , `bytepiece_cli`bin, `bytepiece_py`python lib三个部分
## 算法原理
- https://kexue.fm/archives/9752
- https://kexue.fm/archives/9768
## bytepiece-cli
```bash
# train with default
bytebiece-cli -i train_file -o model_file
```
### cli 相关参数
```text
Usage: bytepiece-cli.exe [OPTIONS] --input-file <INPUT_FILE> --output-file <OUTPUT_FILE>

Options:
      --order <ORDER>
          Order of the model (default: 6) [default: 6]
      --max-vocab-size <MAX_VOCAB_SIZE>
          Maximum vocabulary size (ignored if max-vocab-size-array is set) [default: 10000]
      --max-vocab-size-array <MAX_VOCAB_SIZE_ARRAY>
          Array of vocabulary sizes (comma-separated list, e.g., "8000,16000,32000")
      --max-piece-len <MAX_PIECE_LEN>
          Maximum piece length (default: 36) [default: 36]
      --min-count <MIN_COUNT>
          Minimum count for a piece (default: 2) [default: 2]
      --max-norm-len <MAX_NORM_LEN>
          Maximum norm length (default: 10000) [default: 10000]
      --isolate-digits
          Whether to isolate digits (default: false)
      --ensure-unicode
          Ensure Unicode validity (default: true)
      --workers <WORKERS>
          max workers for parallel training if value greater than 1 [default: 1]
      --batch-size <BATCH_SIZE>
          batch size for parallel training if workers > 1 [default: 100]
  -i, --input-file <INPUT_FILE>

  -o, --output-file <OUTPUT_FILE>

  -h, --help
          Print help
  -V, --version
          Print version

```

## bytepiece-py
- install 
> pip install pybytepiece-xxx.whl
- usage

```python
from bytepiece_py import Tokenizer
tokenizer = Tokenizer.from_json("xxx.model")
tokenizer.tokenize("我是bytepiece分词器", -1.0)
```