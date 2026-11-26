# 準備・スイッチの検証

実行コマンド: `python LOCAL_EXPERIMENT/scripts/flow_study/preflight.py`。所要時間192.1秒。8組合せ、各1epoch（train/val各4系列・2batch）、保存checkpointを確認。disabled branchの画像読み込みは禁止関数で確認。validationはeval、終了時trainに復帰。

既定値はW240×H426・全train動画・seed0。縮小はW120×H213、train動画をID昇順の先頭ceil(25%)、valは全量。入力順序、hpa重複、ConvLSTM5frame、RGBA PNGは変更しない。

Baseline checkpoint・同じseedで改修前後の全出力を比較：最大差0。Baseline snapshot `15221374aa065429d360c139b806cf79b34b49bb`（D1の訓練時点はgit導入前）。

改修diffはshared_switches.diffとmodalities.diff。gitはlocalのみ、remoteなし。

Commit ID: `fa962404d7f18bee4b47084d0b8ffb4efcc8dfa0`。評価はvalidationのみ。testは読み込み・評価しない。

TITANのダウンロード準備は実際のリンク待ち。smoke完了はJAAD準備の完了であり、TITAN取得完了を意味しない。現時点でdisk648GiBの空きだが、パッケージ一覧取得前には容量十分と断定しない。
