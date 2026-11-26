# 公開コードの再現（既存の test 結果）

修正前コード・test 上の過去の結果であり、修正後の validation 比較と直接比較できない。既存 test_results JSON を読むだけで転記した。再学習・再評価はしていない。F1 は標準の横断 F1。author_nonstandard_f1 は作者の別式であり F1 として比較しない。全4runは seed0・22epoch。0.5/1.0/1.5秒=15/30/45出力、入力16。

学習率0.001（論文0.0001）、batch2、workers8。画像W240×H426、RAFT入力/PNG W960×H540 RGBA。元画像は原寸のまま保持した。split の train/val は公開リストを使用。test データを今後の開発比較に再利用しない。
