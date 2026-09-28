# stemcue

日本語 | [English](README.en.md)

曲に合わせて映像を切り替える「音ハメ」のために、曲のどこでカットすればよいかを調べるコマンドラインツール。

曲を渡すと、次のものを時刻付きで一覧にする。

- 拍（曲のテンポを刻む一定間隔の点）と、小節の頭（各小節の 1 拍目）
- ドラムのキック・スネア・シンバルが鳴った瞬間
- 複数の楽器がそろって強く鳴った瞬間
- 楽器が鳴り始める所、鳴り止む所、演奏全体が止まる所

一覧は、拍と小節の目盛り（以下、グリッド）の上に並べる。「このヒットは 12 小節目の 3 拍目の少し前」のように位置が分かるので、カットを小節の頭に置くか、ヒットの瞬間に置くかを決めやすい。

## 何を入力するか

主な対象は、AI 作曲サービスの Suno から書き出した **stem**。stem は、ボーカル、ドラム、ベースのように楽器ごとに分かれた音声ファイルのこと。楽器ごとに分かれていると、どの楽器がいつ入ってきたかまで分かる。

stem がない曲でも、曲全体の音声ファイル 1 つを渡せば動く。ただし精度は落ちる。どのくらい落ちるかは実際の曲で測ってある（[stem と 1 ファイルの違い](#stem-と-1-ファイルの違い)）。

## 中で使っているもの

- **拍と小節の頭**: [beat_this](https://github.com/CPJKU/beat_this) を使う。beat_this は、オーストリアの JKU Linz（ヨハネス・ケプラー大学リンツ）の研究グループが公開している、曲から拍と小節の頭を見つける AI モデル。
- **それ以外（ドラムの打音、楽器の入りと抜けなど）**: 音声解析の定番ライブラリ [librosa](https://librosa.org/) で求める。

beat_this のコードは stemcue の中に取り込んであり、beat_this の学習済みデータ（以下、weight）の読み込みには簡易的なセキュリティ対策をしている。読み込む前にハッシュ値を照合し（[weight とハッシュ値について](#weight-とハッシュ値について)）、数値のデータだけを取り出す専用の処理で読んで、safetensors という形式に変換してから使う。

## できあがるもの

- `cues.json`: 見つけたものすべてを時刻付きで書いたファイル。プログラムや AI エージェントが読む用。
- `viewer.html`: ブラウザで開くと曲が再生され、見つけた打音や拍を楽器ごとの行に並べて表示する。このファイル 1 つで完結していて、ネットにはつながらない。
- `diagnose` と `cuts` の 2 つのコマンドは、結果を JSON（プログラムが読みやすい文字データの形式）で画面に出す。AI エージェントに作業を任せるときに使う。

さらに、Claude Code（Anthropic の AI コーディングツール）用の「スキル」も同梱している。スキルは作業手順書のようなもので、Claude に stemcue の使い方と、出てきたグリッドの誤りを見つけて直す手順、カット時刻の選び方を教える。

## インストール

[uv](https://docs.astral.sh/uv/)（Python のツールを入れて動かすためのソフト）が必要。Python 3.12 が入っていなければ uv が用意する。AI モデルを動かす PyTorch も一緒に入るので、最初のインストールはダウンロードに時間がかかる。

```bash
uv tool install git+https://github.com/nemalabs/stemcue
stemcue weights fetch          # beat_this の weight（final0）をダウンロードして変換する
```

weight は `~/.cache/stemcue/weights` に保存される。置き場所を変えたいときは、`weights` と `analyze` に `--weights-dir` を付ける。`weights fetch` を先に実行しなくても、初めて `analyze` を実行したときに自動でダウンロードする。

### weight とハッシュ値について

stemcue 自体に weight は入っていない。`stemcue weights fetch` は、beat_this の作者が公開しているサーバ（`cloud.cp.jku.at`）から weight（`final0`、`final1`、`final2` の 3 種類）をダウンロードする。ダウンロードしたファイルの SHA-256 ハッシュ値が、`src/stemcue/weights.py` に記録した値と 1 文字でも違えば、そのファイルは使わない。

**記録してあるハッシュ値は、2026-09-28 にダウンロードしたファイルから計算した値で、beat_this の作者が公表した値ではない。** 照合して分かるのは「2026-09-28 時点のファイルと同じものが届いた」ことだけで、そのファイルが作者の意図した正しい配布物だという保証にはならない。作者がサーバ上のファイルを差し替えた場合は、ハッシュ値が合わなくなり `fetch` は止まる。別の経路で weight を手に入れた場合は、中身を自分で確かめてから、ハッシュ値を指定して登録する。

```bash
stemcue weights import FILE --name NAME --sha256 HEX
```

weight の利用条件は beat_this の作者が決めるもので、このリポジトリのライセンスは及ばない。

## 使い方

### 1. 曲を解析する

```bash
# stem を入れたフォルダを渡す場合。全楽器が混ざった完成版の音声（ミックス）もあれば --mix で渡す。
# beat_this はミックスで学習されているので、stem を足し合わせた音よりミックスの方が拍を正しく取りやすい。
stemcue analyze "My Song Stems" --mix "My Song.wav" --out out

# stem がない場合は、曲の音声ファイルを 1 つ渡す
stemcue analyze "My Song.wav" --out out
```

`out` フォルダに `cues.json` と `viewer.html` ができる。`viewer.html` はブラウザで開く（macOS なら `open out/viewer.html`）。

### 2. グリッドが正しいか確かめて直す

beat_this が出すグリッドは、実際の曲ではそのまま使えないことが多い。よくある誤りは次のとおり。

- 小節の頭を多く打ちすぎる
- 曲の途中から、テンポを実際の半分と取り違える
- テンポの揺れるイントロに、無理に一定の拍を当てはめる

そこで、次のコマンドで怪しい所を探して直す。

```bash
stemcue diagnose out/cues.json                          # 怪しい所の一覧（テンポを半分や倍に取り違えた区間、拍の数がおかしい小節、図で見ておくべき時間帯）
stemcue overview out/cues.json --out out/overview.png   # 曲全体について、楽器ごとの音量の変化とグリッドを 1 枚の図にする
stemcue look out/cues.json --start 28 --end 40 --out out/look_28.png   # 指定した時間帯（最大 60 秒）を拡大した図
stemcue grid out/cues.json --fix fix.json --out fixed   # 補正ファイルに従ってグリッドを作り直す
```

補正ファイル（`fix.json`）は、曲の区間ごとに拍の置き方を書いた JSON。たとえば「0〜16 秒はテンポが自由なので拍を置かない」「27.5 秒からの 1 小節だけ 6 拍」「30 秒以降は beat_this の拍を使い、外れた点を除いてならす」のように書く。書き方と、図の読み方、直すときの判断基準は、スキルの手順書 [`SKILL.md`](.claude/skills/stemcue/SKILL.md) に詳しく書いてある。Claude Code を使わなくても、手順書として読める。

### 3. カットの候補を出す

```bash
stemcue cuts fixed/cues.json --fps 24
```

動画のフレームレート（1 秒あたりのコマ数。例では 24）を指定すると、カットの候補をフレーム番号付きで出す。候補は次の 3 種類。

- 小節の頭
- セクションの頭（複数の楽器が入る・抜ける、音量が大きく変わる、止まっていた演奏が再開する、のどれかに当たる小節の頭）
- 強いヒット（小節の何拍目のどのあたりかも付く）

### 出力と終了コード

どのコマンドも `--help` を付けると説明が出る。結果は標準出力に出る。`analyze`、`grid`、`look`、`overview`、`weights` は「項目名、タブ、値」の行で出し、`diagnose` と `cuts` は JSON で出す。途中経過は標準エラー出力に出る。

終了コードの意味は次のとおり。

| コード | 意味 |
|---|---|
| 0 | 成功 |
| 2 | コマンドの使い方の誤り |
| 3 | weight の検証に失敗した |
| 4 | 入力の不備（音声、`cues.json`、補正ファイル） |
| 5 | ネットワークの失敗 |

## stem と 1 ファイルの違い

stem がないと、どの音がドラムかをファイルから直接は知ることができない。そこで、曲の音を「伸びる音」と「叩く音」に分ける手法（librosa の HPSS）で打楽器の成分だけを取り出し、そこからキック・スネア・シンバルを探す。

精度がどれだけ落ちるかを、Suno の曲 1 曲（147.8 秒）で測った。同じ曲を、stem のまま解析した場合と、stem を足し合わせて 1 ファイルにして解析した場合とで比べている。時刻の差が 30 ミリ秒以内なら同じものを見つけたとみなした。

| 見つけたもの | stem で解析 | 1 ファイルで解析 | stem の結果のうち、1 ファイルでも見つかった数 | 1 ファイルでだけ見つかった数（誤検出の可能性が高い） |
|---|---|---|---|---|
| 拍 | 328 | 322 | 322 | 0 |
| キック | 407 | 502 | 399 | 103 |
| スネア | 428 | 522 | 422 | 100 |
| シンバル | 249 | 252 | 156 | 96 |
| 複数の楽器がそろった強打 | 181 | 333 | 126 | 207 |
| 楽器が鳴り始めた所 | 102 | 3 | — | — |
| 楽器が鳴り止んだ所 | 102 | 3 | — | — |

- 拍はほぼ同じ位置に取れる。
- キックとスネアはほとんど見つかるが、余計なものが 2 割ほど混じる。シンバルは 4 割近くを見落とす。
- 1 ファイルでは、どの楽器が入ってきたか・抜けたかが分からない。そのため `cuts` がセクションの頭を判断する材料は、音量の変化と演奏の停止だけになり、候補が少なくなる。
- 測ったのは 1 曲だけなので、数字は目安として見てほしい。

## librosa だけでは足りない理由

librosa は、音が鳴り始めた瞬間や、高音・低音それぞれの音量を測るのが得意で、stemcue もそのために使っている。ただ、拍と小節を librosa だけで求めようとすると、次の問題がある。

- **小節の頭が分からない。** librosa の拍を探す機能（`beat_track`）は拍の位置しか返さず、どれが小節の 1 拍目かは教えてくれない。小節の頭でカットしたいなら、この情報がないと始まらない。
- **テンポを半分や倍に取り違えやすい。** 上の表と同じ曲の冒頭 30 秒では、librosa はテンポを 94 BPM（1 分間に 94 拍）と判定し、拍の間隔は 0.63 秒だった。beat_this の拍の間隔は 0.32 秒で、librosa は 1 拍おきにしか拍を打っていないことになる。
- **拍の位置がずれる。** 同じ 30 秒で、librosa の拍 48 本のうち 9 本は、beat_this の拍から 40 ミリ秒以上（最大 162 ミリ秒）ずれていた。24 fps の動画では 1 コマが約 42 ミリ秒なので、最大で 4 コマ近いずれになる。

beat_this も完璧ではなく、曲の途中でテンポを半分と取り違えることがある。だから stemcue は、`diagnose` で怪しい所を見つけ、補正ファイルで直す使い方を前提にしている。librosa の拍の検出結果は、beat_this と食い違う区間を見つけるための照合用として `cues.json` に残している（`beat_check`）。

## Claude Code のスキルを入れる

スキルのフォルダを、全プロジェクト共通の置き場か、使いたいプロジェクトの中にコピーする。

```bash
git clone https://github.com/nemalabs/stemcue
cp -R stemcue/.claude/skills/stemcue ~/.claude/skills/                 # すべてのプロジェクトで使う場合
cp -R stemcue/.claude/skills/stemcue /path/to/project/.claude/skills/  # 1 つのプロジェクトだけで使う場合
```

スキルは、上のインストール手順で入る `stemcue` コマンドを使う。Claude Code に stem のフォルダや曲のファイルを渡して「ヒットの位置を調べて」「小節の頭でカットしたい」のように頼むと、スキルの手順に沿って作業する。

## 開発者向け

```bash
git clone https://github.com/nemalabs/stemcue && cd stemcue
uv sync
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src
```

テストは、実際に `stemcue` コマンドを動かして結果を確かめる。weight の変換結果を元のファイルと照らし合わせるため、元の `final0.ckpt` を `.cache/stemcue/download/final0.ckpt` に置いておく必要がある。

```bash
mkdir -p .cache/stemcue/download
curl -fL -o .cache/stemcue/download/final0.ckpt \
  https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt
uv run pytest
```

## ライセンス

stemcue は MIT ライセンス（[`LICENSE`](LICENSE)）。`src/stemcue/vendor/beat_this/` に取り込んだ beat_this のプログラムは、JKU Linz の Institute of Computational Perception による MIT ライセンス（同じフォルダの `LICENSE`）。weight はこのリポジトリに含まれず、利用条件も別（[weight とハッシュ値について](#weight-とハッシュ値について)）。
