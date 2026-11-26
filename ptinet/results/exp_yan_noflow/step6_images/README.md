# 光フローPNGの確認

コマンド: `python LOCAL_EXPERIMENT/scripts/flow_study/images.py`（CPUのみ）。所要時間9.9秒。valから自車moving/stopped×歩行者moving/stationaryを各2例、可能な限り異なる動画を選ぶ。実際の群カバレッジはcoverage.csv。緑枠は対象歩行者の観測初frame bbox、動静ラベルは観測区間に基づく。

白色判定：RGB全channel≥245かつmax−min≤10、alphaは比率計算から除外。RGBA alpha=255も確認。flow_vizはframeごとの相対正規化なので、白色比率は小さな可視化変位の指標で、絶対速度や静止の証明ではない。

実装は公式RAFT FP32 iters20→flow_viz色画像→RGBA alpha255 PNG（960×540）。RAFTは960×540入力をpadし、出力をunpadして同解像度で保存、DataLoaderで4channelとして読み込み既定PIL W240×H426にresizeする。元のwidth/height順序は修正しない。

指令に記載された論文設定：RAFT、240×420。本実装との相違は保存/計算解像度960×540、モデル入力の実形状H426×W240、4channel可視化PNG。論文の生flow channel数/正規化等は提示資料から未確認であり推測しない。

Commit ID: `fa962404d7f18bee4b47084d0b8ffb4efcc8dfa0`。評価はvalidationのみ。testは読み込み・評価しない。
