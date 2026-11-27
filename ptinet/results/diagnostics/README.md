# 修正の根拠（既存の validation 診断）

修正前の checkpoint を validation で調査した集計のみ。新しい学習・test 評価は行っていない。intention_val_diag は確率・閾値、intention_relu_check は softmax 前の線形出力を調査。0.5 秒の全検体で横断側 z1≤0、ReLU により当該側が0になり、横断確率は0.5を超えなかった。validation を eval にする修正と、この ReLU の削除を適用した。診断そのものは修正前モデルの結果で、修正後 D1/D2 の性能値とは別。raw sample・checkpoint・ログは含まない。

元の README にある LOCAL_EXPERIMENT は当時のローカル作業領域を表す。掲載していない詳細ファイルへの参照は履歴説明であり、実行入口ではない。
