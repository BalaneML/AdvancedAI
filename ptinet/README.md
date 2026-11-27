# Corrected PTINet Baseline

## 1. Quickstart

NVIDIA CUDA の例。Python3.12、git、wget、unzip と対応GPUドライバを用意する。ROCm/WSL の場合は手順2を `bash scripts/setup_env.sh rocm-wsl` に置き換える。

1. `git clone --branch baseline-v1 https://github.com/BalaneML/AdvancedAI.git && cd AdvancedAI/ptinet`
2. `bash scripts/setup_env.sh cuda`
3. `HSA_ENABLE_DXG_DETECTION=1 .venv/bin/python scripts/data/prepare_jaad.py --phase all --workers 8`
4. `HSA_ENABLE_DXG_DETECTION=1 .venv/bin/python tools/smoke.py --config configs/small.yml`
5. `bash scripts/download_checkpoints.sh`
6. `HSA_ENABLE_DXG_DETECTION=1 .venv/bin/python tools/evaluate.py --config configs/full.yml --checkpoint checkpoints/D1-baseline-best.pkl --reference results/reference_d1.csv --output runs/reproduced_d1.csv`

手順6は ADE/FDE が基準から各0.05px以内、F1/Accuracy が一致することを確認し、不一致なら停止する。データ・環境を確認してから実験する。D1基準は validation / seed0 / N616、ADE **12.1355**、FDE **23.8779**、F1 **0**、Accuracy **0.8831169**。評価時にもVAEが乱数を使うため seed0 を固定する。作業ツリーに変更がある場合は実験前にcommitする。

## 2. 概要

[著者公開PTINet](https://github.com/munirfarzeen/PTINet) に基づく。位置・速度VAE、LCF、画像ConvLSTM、フローResNetと融合の構成は維持。以下の2つの誤りを修正し、同じ比較手順を全員で使う。著者コードの由来・引用は同リポジトリ参照。JAADは[提供元](https://github.com/ykotseruba/JAAD)の利用条件を各自確認する。

## 3. 修改内容

パスの可搬化、train/val取り違え、PNG読込・DataLoader workers、標準指標出力を修正。validation に入ると `eval()`、終了時 `train()` に戻す。意図出力の softmax 直前の ReLU を除去。F1 は標準式 `2TP/(2TP+FP+FN)`（0除算=0）、系列スコア=max_t P(cross)、厳密な >0.5、Accuracyも系列単位。根拠は [diagnostics](results/diagnostics/README.md)。

共通ツールの縮小は既定で無効（train_video_fraction=1、W240/H426）。安全のため共有runnerは常にtestをスキップする。Baselineでは画像・フロー両方必須。作者内部の use_image/use_opticalflow フィールドはモデル読込の忠実性のため残すが、このブランチではOFFを拒否する。除去実験の公開スイッチはBだけ。

## 4. 既知の問題（未修正）

- resize の幅高さが逆の W240×H426。
- 融合で同じ hpa 特徴を2回加算。
- ConvLSTM は16入力の先頭5フレームだけ使用。
- 光流はベクトルそのものではなく W960×H540 のカラーPNG、RGBA4ch（alpha255）を読む。
- 修正後も0.5秒横断予測はほぼ全て非横断、D1/D2ともF1=0。Accuracyだけで学習成功と判断しない。
- attrib_decoder / fc_attrib 有定義だが forward では未使用。

## 5. 環境搭建

実測はUbuntu/WSL2、Python3.12、RX9070XT、ROCm7.2.1、torch2.9.1、torchvision0.24.0。主要依存はrequirements.txtを固定。NVIDIAは `bash scripts/setup_env.sh cuda`（torch2.9.1/torchvision0.24.1/cu128）、AMD WSLは `bash scripts/setup_env.sh rocm-wsl`（AMD公式wheel・ホストドライバ依存）。CUDA構成はインストール手順を用意したが、このPCで実測していない。

[PyTorch公式](https://pytorch.org/get-started/previous-versions/) と [AMD WSLガイド](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/wsl/install-radeon.html) を参照。AMD WSLではHSA_ENABLE_DXG_DETECTION=1が必要。setupはvenv内のbundled libhsaを退避しホスト側WSL対応runtimeを使用する。システム環境には変更しない。

## 6. データ準備

`prepare_jaad.py --phase download` で公開JAAD/RAFTを pinned commit で取得、JAAD_clips.zip と公式raft-things weightsをdownload。`--phase frames --workers 8` は原寸動画を0始まり5桁PNGにcv2で全フレーム抽出（動画ごと並列、各decoder1thread）。`--phase annotations` は公開したtrain242/val69動画IDのみCSV生成。`--phase flow` はCSVに必要な各フレームを公式RAFT-things FP32/iters20で処理。t→t+1、最終フレームは(t−1,t)。入力960×540、flow_viz RGBにalpha255を追加しRGBA PNG同サイズ、4読込/4書込スレッド・中断再開可能。testデータは生成・評価しない。

配置：`data/JAAD/{JAAD_clips,annotations,images,PN_ego/{train,val},opticalflow}`、`third_party/RAFT/{core,models}`。データはgit管理対象外。公開train/valリストとtrain先頭ceil(25%)=61動画リストはsplits/。CSVは絶対パスを生成するので移動時はannotationsを再生成する。full設定の読込順もsplits/*_loading_order.txtに固定し、ファイルシステムの列挙順でVAE乱数と系列の対応が変わらないようにした。動画の分割・学習内容は変わらない。

初回の公開動画archive約3.1GB（回線依存）。過去346動画82032原寸PNGで175.5GB・抽出約9分（8並列）、全67731必要フローの200枚実測換算は約1.92時間。今回はtrain/val311動画なのでこの全量値は上限の目安。archive、画像、フロー、複数runを含め500GB程度の空きを推奨。これらはデータの目安で、repo配布サイズではない。

## 7. ハードウェア要件

RX9070XT 16GB、Ryzen7 7700で確認。D1 full/batch2のPyTorch peak allocated は約12.28GB（11.43GiB）。整カード総使用量とは違い、他のGPUアプリは終了してから実行する。VRAM不足時は configs/small.yml を使用し、縮小結果はfullと混ぜない。

## 8. 実行時間

RX9070XT、seed0、22epoch。D1 fullは約8.42時間（epoch約23分）、再学習noflowは6.36時間。縮小設定の軌跡のみ4.44分・軌跡+flow16.66分は画像/LCFなしの別モデルで、full構造の縮小Baseline全学習時間の実測値ではない。縮小Baselineは冒煙1epochの結果を公開確認表に記録する。未測定のfull構造small22epochの時間は断定しない。

## 9. よくある問題

WSLの `rocm-smi` がamdgpu未初期化でもtorch GPUが動く場合がある。まず `HSA_ENABLE_DXG_DETECTION=1 .venv/bin/python -c 'import torch; print(torch.cuda.is_available())'`。DataLoader待ちならloader_workers:8（本機で改善）。RAM/IOが不足するPCはworkersを調整する。OOMならsmall設定、GPU競合を確認。環境セットアップは新venvで行い既存実験環境のtorchを上書きしない。PNG mode/解像度不一致や基準誤差は原因調査してから続行する。

## 10. 比較ルール

開発比較はvalidationだけ、testは最後に残す。ADE/FDE(px)、標準F1、Accuracy、Latencyを記録。Latencymodelはbatch1/10warmup/100samples中央値（入力用意・GPU転送を除外、同期あり）。`python tools/benchmark.py --config configs/full.yml --include-raft --output runs/timing.csv` はRAFT別計測と和の参考推定も出す。公開スクリプトはRAFTも10warmup/100件（過去の参照表はRAFTのみ5warmup/50件）。model LatencyとRAFTを含む推定は列を分け、後者を実測と呼ばない。公平なLeaderboard主列はmodel Latency、RAFT込みはメモ欄。同PC・同設定Baselineを測り、短縮率=100×(Baseline−実験)/Baseline。異なるPCで以前の絶対時間を流用しない。僅差なら複数seed。seed/commit/PC/full-smallを必ず記録。

## 11. 開発フロー

分岐→push→Leaderboard記録→Issueで全員のフィードバック→採否→PR→実装者以外の全員承認→最後の承認者がmerge。mainへ直接pushしない。不採用の枝も削除しない。このBaselineはmainへのPR、exp/yan-noflowはIssueだけ。

## 12. 自分の実験を追加

`exp/<名前>-<内容>`（例 exp/oda-xxx）をbaseline-v1から作る。既定OFFスイッチで実装しBaselineを変えない。設定はconfigs/exp/、結果はresults/。`python tools/train.py --config configs/full.yml`（testは常にskip）、評価とbenchmarkを同じ版で実行。コード変更は実行前にcommit。例は exp/yan-noflow のREADME_exp_yan_noflow.md。leaderboard_template.csvの日本語列をそのまま使用する。

## 13. Baselineの版

baseline-v1。将来の共通変更は審査PRを経てbaseline-v2等の新タグを付け、比較対象の版もLeaderboardメモに残す。既存結果のtraining commitは書き換えない。D1はpre-git＋[snapshot](results/code_snapshots/README.md)。公開版コードIDとの対応はresults/provenance.json。

## 14. 改良の判断基準

グループで決定。候補：Latencyが短縮し、ADE/FDE変化がseed間の変動範囲以内。F1/Accuracyを含めトレードオフを確認し、単seedの微差を精度改善と呼ばない。

## Release と関連結果

[baseline-v1](https://github.com/BalaneML/AdvancedAI/releases/tag/baseline-v1) のD1/D2をcheckpoints/へ置く。ファイル名、bytes、SHA256は [checkpoint_assets.json](results/checkpoint_assets.json)。download_checkpoints.shはSHAとbytesを自動検証する。[修正前のtest再現](results/reproduction/README.md) はvalidationと直接比較不可。[D1/D2](results/corrected_d1_d2.csv) と [再現基準](results/reference_d1.csv)。
