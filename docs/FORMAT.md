# NCW1 텐서 파일 포맷

`nc_save` / `nc_load` (C), `save_tensors` / `load_tensors`, `save_weights` / `load_weights` (Python)가 쓰는 포맷입니다.
모든 정수는 리틀 엔디언입니다.

```
offset  크기        내용
0       4           매직 "NCW1"
4       u32         버전 (1)
8       u32         엔트리 수 n
12      ...         엔트리 n개

엔트리:
  u32               이름 길이 L (바이트, 최대 4096)
  L                 이름 (UTF-8, 널 문자 없음)
  u32               dtype: 0 = f32, 1 = i32, 3 = i8
  u32               차원 수 d (최대 6)
  i64 × d           shape
  u64               데이터 바이트 수 B (= 원소 수 × dtype 크기)
  B                 데이터 (row-major, 연속)
```

- 이름은 `state_dict()` 키입니다 (`0.weight`, `1.running_mean` 등, PyTorch와 같은 규칙)
- 읽는 쪽은 요청하지 않은 엔트리를 건너뛰므로, 파일에 추가 텐서가 있어도 됩니다
- `nc_load`는 이름이 없거나 dtype/shape가 다르면 에러를 냅니다. Python `load_weights`는 `load_state_dict(strict=True)`와 같은 규칙입니다
- 양자화된 `Linear`는 `weight`(i8 `[out, in]`), `scale`(f32 `[out]`), `bias`(f32 `[out]`)로 저장됩니다
