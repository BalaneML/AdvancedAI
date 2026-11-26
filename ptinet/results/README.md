# 結果と provenance

baseline.csv は公開版の評価コードを実行した D1 validation の値。reference_d1.csv は公開版を検証するための既存 flow_study 基準値（seed0、616系列）。ADE/FDE の許容差は各0.05px、F1/Accuracy は一致。code commit と、その後の文書だけの commit を分けて記録する。checkpoint の学習は pre-git なので公開 commit を学習 commit と呼ばない。

corrected_d1_d2.csv は既存の corrected 診断集計。D2 は LCF OFF で、ADE12.179165/FDE23.825284/F1=0。reproduction は修正前 test、diagnostics は修正前 validation の意図診断。preliminary profiler/masking/FLOPs/noise は除外した。

benchmark_reference.csv は過去の同一PCでの規定プロトコル測定。benchmark_publication.csv は今回の公開版での確認であり、GPU クロック等で絶対時間が変わり得る。比較には同じPC・設定・手順を使用する。
