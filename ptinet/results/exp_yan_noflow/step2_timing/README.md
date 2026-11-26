# 推論時間の実測

コマンド: `python LOCAL_EXPERIMENT/scripts/flow_study/timing.py`。所要時間76.1秒。

計測条件: {'computer': 'DESKTOP-E8L0GNB', 'system': 'Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39', 'gpu': 'AMD Radeon RX 9070 XT', 'torch': '2.9.1+rocm7.2.1.gitff65f5bc', 'hip': '7.2.53211-e1a6bc5663', 'batch': 1, 'warmup': 10, 'measured': 100, 'cpu_model': 'AMD Ryzen 7 7700 8-Core Processor'}。同じval100系列（10series warm-up）、batch1、eval/no_grad、cuda synchronize前後、中央値とP90。全モデルが横断と軌跡の両出力を返すまで。実験checkpointは読み込まず、作者constructorのResNet ImageNet初期化を保持し、その他はseed0初期化。入力テンソルの準備、読込、転送、RAFTはモデル推論時間に含まない。順番は各サンプルで回転。

RAFTは公式raft-things.pth、FP32、iters20、各解像度5回warm-upと50対val隣接フレーム。実測はGPU推論のみ。960×540は960×544、426×240は432×240にpad。前段Aの再利用workerで当日再計測、旧結果の流用ではない。

主指標はモデル推論時間。補足の端到端推定は1frameあたりRAFT1回＋観測窓全体forward1回の中央値和で、IO等は含まずオフライン事前計算時にはRAFT不要。モデルパラメータ数は重み総要素数、未使用attrib層を含め実モデル全体を数える。

本体画像の既定リサイズはPIL W240×H426のまま。attrib_decoder / fc_attrib 有定义，但 forward 中未使用。

Commit ID: `fa962404d7f18bee4b47084d0b8ffb4efcc8dfa0`。評価はvalidationのみ。testは読み込み・評価しない。
