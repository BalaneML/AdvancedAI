# 軌跡のみ vs 軌跡＋光フロー

summary.csvに2モデルの実測結果を併記。両方seed0・22epoch・train先頭25%動画・val全量、LCF/画像を使用しない。各runのconfig/complete.jsonに所要時間を記録。光フロー追加で改善すれば本学習条件で追加情報を利用した証拠となるが、差が小さくても光フローに情報が全くないことの証明にはならない。画像座標と小規模学習の制約を残す。

Commit ID: `fa962404d7f18bee4b47084d0b8ffb4efcc8dfa0`。評価はvalidationのみ。testは読み込み・評価しない。
