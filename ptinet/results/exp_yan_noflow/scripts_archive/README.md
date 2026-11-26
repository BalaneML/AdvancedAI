# 過去の実験スクリプト（方法の控え）

LOCAL_EXPERIMENT は当時の非公開作業領域の置換。これはそのまま実行するQuickstartではない。新しい環境では tools/train.py、tools/evaluate.py、tools/benchmark.py、tools/mask_inputs.py と configs/exp/ を使用する。過去の固定設定・生サンプル・ログ・巨大データは配布しない。pair統計の元サンプルは再評価時に --samples-output で保存できる。数値は既存のCSVをそのまま転記した。

修正前モデルでも予備実験を行い、光フロー寄与が小さいという方向は一致した。数字が異なるため、修正前のprofiler・FLOPs・masking・noise表はここには含めない。
