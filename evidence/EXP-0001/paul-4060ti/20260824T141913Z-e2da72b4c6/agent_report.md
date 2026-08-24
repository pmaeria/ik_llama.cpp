# EXP-0001 Result

- Status: **complete**
- Target: **Qwen3.6-35B-A3B**
- Machine: `paul-4060ti`
- Source commit: `e2da72b4c6acf05053f17bdbed6c58d0b313a927`
- Model: `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`
- Model SHA-256: `0d17e255dc257a11f398ed4bc8d62412d8ce9ca24b3fce2947d962e4bfed5758`

| Config | Test | Median t/s | Min | Max |
|---|---:|---:|---:|---:|
| `fit-default` | `pp512` | 225.599 | 223.208 | 227.991 |
| `fit-default` | `tg256` | 29.888 | 29.802 | 29.975 |
| `fit-default` | `tg256@pp32768` | 27.926 | 27.909 | 27.944 |
| `fit-default` | `tg256@pp8192` | 29.268 | 29.208 | 29.328 |
| `fit-no-active-only` | `pp512` | 227.524 | 226.675 | 228.372 |
| `fit-no-active-only` | `tg256` | 29.880 | 29.761 | 29.999 |
| `fit-no-active-only` | `tg256@pp32768` | 27.917 | 27.793 | 28.041 |
| `fit-no-active-only` | `tg256@pp8192` | 29.390 | 29.386 | 29.394 |
| `fit-no-fused-moe` | `pp512` | 221.635 | 220.658 | 222.612 |
| `fit-no-fused-moe` | `tg256` | 24.883 | 24.824 | 24.941 |
| `fit-no-fused-moe` | `tg256@pp32768` | 23.357 | 23.213 | 23.501 |
| `fit-no-fused-moe` | `tg256@pp8192` | 24.577 | 24.227 | 24.928 |
| `fit-no-graph-reuse` | `pp512` | 227.817 | 225.019 | 230.615 |
| `fit-no-graph-reuse` | `tg256` | 29.180 | 29.100 | 29.260 |
| `fit-no-graph-reuse` | `tg256@pp32768` | 27.067 | 27.028 | 27.106 |
| `fit-no-graph-reuse` | `tg256@pp8192` | 28.360 | 28.242 | 28.478 |
| `fit-q8kv` | `pp512` | 304.933 | 294.087 | 315.778 |
| `fit-q8kv` | `tg256` | 37.933 | 37.915 | 37.952 |
| `fit-q8kv` | `tg256@pp32768` | 29.925 | 29.798 | 30.052 |
| `fit-q8kv` | `tg256@pp8192` | 35.967 | 35.893 | 36.041 |

Correctness: **pass**

- `short_chat.txt`: pass `b3679c461e41cd72b6ee15be1857082fa374e6a91d95df3df7c0c8d07096be66`
- `tool_json.txt`: pass `e0b99e6f1c33ccfb3097525ab920d9ee9ca9166e3c938c1970dc5e9e8c3fcaf7`
