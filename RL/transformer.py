"""
分類機を複数設定することによる強化学習。行動を直接出力するモデルになる。
エージェントの行動がたくさんある場合について強そう。近傍の行動が定義しやすいと尚よい。
環境の変数は複数のトークンで生成され、行動も同様にトークンで生成される。
モデルは以下の三つ。
[A]env_tokens -> action_tokens
[B]env_tokens + action_tokens -> value

1. playの際
Aモデルでアクション(a)を生成。そこから近傍のアクション(a_)を作成。
Bモデルで優劣を判断し焼きなましの要領で行動を決定していく。
（この際、学習の進行度に応じて作成する近傍アクションの遠さを操作しても良いか）
2.learningの際
Aモデルについては選択された真のaction_tokenとの誤差を縮めるように
BモデルについてはvalueをAdvantageに近づけるように。
"""