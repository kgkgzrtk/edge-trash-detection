# Trash Detection Dataset

## 概要
このデータセットは、ゴミ検出のために設計されており、2023年7月7日にRoboflowを通じて提供されました。データセットはCC BY 4.0ライセンスの下で提供されています。

データセットの詳細は以下のリンクから確認できます：
[Trash Detection Dataset](https://universe.roboflow.com/trash-dataset-for-oriented-bounded-box/trash-detection-1fjjc)

## ライセンス
このデータセットは、クリエイティブ・コモンズ 表示 4.0 国際 (CC BY 4.0) ライセンスの下で提供されています。詳細なライセンス情報は、以下のリンクから確認できます：
[CC BY 4.0 ライセンス](https://creativecommons.org/licenses/by/4.0/)

## 利用ケース
このプロジェクトのいくつかの利用ケースを以下に示します：

1. **スマート廃棄物管理**：
   - ゴミ検出は、都市や自治体がスマート廃棄物管理を行うのに役立ちます。例えば、ゴミ箱内の廃棄物の自動分類、ロボットによる選別、より頻繁なゴミ収集が必要なエリアの特定などです。

2. **公共意識向上キャンペーン**：
   - 政府や環境団体は、公共の場で見つかる一般的なゴミの種類を特定し、適切な廃棄方法を促進するために、ゴミ検出モデルを使用して公共意識向上キャンペーンを作成できます。

3. **ビーチクリーンアップロボット**：
   - ゴミ検出モデルは、ビーチ清掃ロボットに実装され、ゴミをより効果的に検出し収集することで、人間や海洋生物にとってより清潔で安全な海岸を提供します。

4. **ゴミ報告用モバイルアプリ**：
   - ゴミ検出モデルを使用したアプリを作成し、市民が地域のゴミを報告・記録できるようにします。収集されたデータは、地域の清掃活動、問題エリアの特定、廃棄物管理の改善提案に利用されます。

5. **産業および製造工場**：
   - ゴミ検出モデルは、産業や製造工場でのリサイクルおよび廃棄物管理に役立ち、リサイクルおよび廃棄プロセスの最適化、コスト削減、環境負荷の軽減を可能にします。

## データセットの詳細
- データセットには6783枚の画像が含まれており、COCO形式でアノテーションされています。
- 各画像には以下の前処理が適用されています：
  - ピクセルデータの自動方向付け（EXIF方向の削除）
  - 416x416へのリサイズ（ストレッチ）

- 各画像には以下の拡張が適用されています：
  - 50%の確率で水平反転
  - 50%の確率で垂直反転
  - 90度回転（なし、時計回り、反時計回り、逆さま）のいずれかを等確率で適用
  - 画像の0〜25%をランダムにクロップ
  - -25度から+25度のランダム回転
  - 水平および垂直に-15°から+15°のランダムシアー
  - -20%から+20%のランダムな明るさ調整
  - -20%から+20%のランダムな露出調整
  - ピクセルの2%に塩と胡椒ノイズを適用

## 参考リンク
- [Roboflowの公式GitHubリポジトリ](https://github.com/roboflow/notebooks) - 最新のコンピュータビジョンのトレーニングノートブックを利用できます。
- [Roboflow Universe](https://universe.roboflow.com) - 100,000以上のデータセットと事前学習済みモデルを見つけることができます。