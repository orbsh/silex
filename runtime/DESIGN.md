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
- 中文分词如需 tokenizer.json 级方案（预训练编码器路线，见下节），另立编译通道——当前 LightGBM 路线用 char n-gram TF-IDF，无需分词器
- 热更新：新资产过 gate 后原子 rename 覆盖 `model.onnx`，Session 按需重建

## 编码器路线（第 ③ 段替换，未实现）

预训练编码器的在线形态与现有链路同构，只换掉特征段（规格见 `encoder-route.md`）：

```
输入文本 batch
  ↓ ① L4: stats_ops 解释执行（同现有实现）
  ↓ ② L3: encoder.onnx（tokenizer.json 随资产交付 → token_ids/attention_mask → 池化 mean/cls）
  ↓ ③ 拼接 [dense | stats]（沿用现有列序契约）
  ↓ ④ L2: 线性头或现有模型（形 B）／编码器图内已含分类头（形 A，单 session 出 probabilities[:,1]）
  ↓ ⑤ 阈值判定 + 低置信带采样回流
```

- 形 A（头在图内）：manifest 增 `tokenizer` / `max_len` / `pooling` 字段，`score` / `threshold` 语义不变
- 形 B（冻结编码器当 L3）：`encoder.onnx` 占现在 `embed.onnx` 的位置，L2 与门禁完全不动，迁移成本最低
- 导出限制：编码器 ONNX 须关 flash attention（eager），unpadding 丢失；现行 batch=1 单请求形态不受影响（unpadding 是批内不等长的优化，吃它好处的是训练吞吐）
- 交付精度默认 int8 动态量化：149M 级编码器约 127MB / 短序列 CPU 约 10ms 量级，热替换体积与现有 50MB 级不同量级
