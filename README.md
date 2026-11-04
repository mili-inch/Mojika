# Mojika [文字禍]

文章・画像・手描きヒントから256×256の漢字画像を生成するモデルです。

## セットアップ

Linux x86_64、Python 3.13、[uv](https://docs.astral.sh/uv/)、Node.js 22.6以上、CUDA 13.0用PyTorchが動くGPUで動作確認しています。
学習・データ準備・デモにはbfloat16対応GPUを使います。RTX 5090（32GB）で確認しています。

リポジトリのルートで実行します。

```bash
uv sync --locked
cd web
npm ci
npm run build
cd ..
uv run --locked python -m mojika.assets
```

最後のコマンドでMojikaの重み（約538MB）と、条件付けに使うSarashina・CLIPをダウンロードします。

## 生成デモ

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked python -m mojika.server --config configs/studio.toml
```

[http://127.0.0.1:8000](http://127.0.0.1:8000) を開き、モデルの読み込み後に編集ボタンを押します。GPT文章、CLIP文章／画像、手描きヒントを入力して生成できます。

## 学習

フォント（例: Noto Sans JPのTTF）と、`{"山": ["高く盛り上がった地形。"]}` 形式の説明文JSONを用意します。
`examples/descriptions.json` に動作確認用の16文字の例があります。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked python -m mojika.prepare_data \
  --font /path/to/NotoSansJP-Regular.ttf \
  --descriptions examples/descriptions.json --output-dir assets/dataset

uv run --locked python train.py stable \
  --config configs/train.toml --dataset-root assets/dataset \
  --output-dir outputs/stable --max-steps 100000
```

データ準備は画像、Sarashina／CLIP埋め込み、条件平均を作成します。データは `images/U+XXXX.png`、`sarashina_embeddings/U+XXXX_N.safetensors`、`clip_embeddings/U+XXXX_N.safetensors`、`condition_means.safetensors` の構成です。
既存のデータ・学習出力ディレクトリは上書きしません。学習はCtrl+Cで現在のバッチ終了後に保存します。
`configs/train.toml` はバッチ16、`configs/published.toml` は配布重みの学習設定（バッチ128）です。バッチサイズは8の倍数で、学習データは最低1バッチ分必要です。

```bash
# 再開。max-stepsは累計ステップ数。
uv run --locked python train.py resume --run-dir outputs/stable --max-steps 200000

# Warmup完了後のチェックポイントからDecayを分岐。
uv run --locked python train.py decay \
  --checkpoint outputs/stable/latest.pt --output-dir outputs/decay \
  --decay-steps 50000 --curve cosine --min-lr-ratio 0.01

# 推論用にモデル・条件付けの重みだけを書き出す。
uv run --locked python -m mojika.conditional.export \
  --checkpoint outputs/decay/latest.pt --output outputs/weights.pt
```

自分で学習した重みを使うには `configs/studio.toml` の `checkpoint` を変更します。
配布モデルは134,399,488パラメータ、13,697文字、200,000ステップ＋Decay 50,000ステップで学習しています。

## 構成・検証

`mojika/conditional/` がモデル・学習部品、`train.py` が学習CLI、`mojika/server/` と `web/` がデモです。

```bash
uv run --locked ruff check mojika tests train.py
uv run --locked python -m unittest discover -s tests -v
cd web
npm run lint
npm test
npm run build
```

PythonテストはCPUで実行できます。GPUでの短時間学習・再開・Decay・書き出しは、データ準備後に `uv run --locked python -m tests.smoke_training --dataset-root assets/dataset --output-dir outputs/smoke` で確認できます。

[MIT License](LICENSE)。第三者コード・フォント・外部エンコーダの条件は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
