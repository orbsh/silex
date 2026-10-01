# Rust 在线塔基（设计文档，代码未交付）

装载契约 = `dist/manifest.json`。推理链重放顺序（与 `skill/scripts/lib.py::predict_dist` 语义一一对应，训练参照即在线规格）：

```
输入文本 batch
  ↓ ① L4: stats_ops 解释执行（char_len / char_count:<c> / keyword_hit:<kw>，纯 Rust，<10µs）
  ↓ ② L3: embed.onnx → ort Session #1（StringTensor 入口，[batch,1]，<50µs 级）
  ↓ ③ 拼接 [dense | stats]（manifest.feature_dim 校验列序）
  ↓ ④ L2: model.onnx → ort Session #2（FloatTensor，1~3ms，只请求 probabilities 输出）
  ↓ ⑤ 阈值判定（manifest.threshold）+ 低置信带采样回流传入飞轮
```

要点：
- `ort = "=2.0.0-rc.13"`（v2 API：`Session::builder()?.commit_from_file()?`，`ort::inputs!`；蓝图的 v1 写法 `.unwrap()` 链已过期）
- 备选纯 Rust 运行时：`tract`（免 ONNX Runtime 原生库）
- 中文分词如需 tokenizer.json 级方案（微型 BERT 路线），另立编译通道——当前 LightGBM 路线用 char n-gram TF-IDF，无需分词器
- 热更新：新资产过 gate 后原子 rename 覆盖 `model.onnx`，Session 按需重建
