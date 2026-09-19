Semlf-RyzenAI
=============
https://github.com/Yasei-no-otoko/Semlf-RyzenAI
Fork of https://github.com/TheoLeeCJ/SemIf

日本語
------
AMD Ryzen AI 1.8.0と公式Qwen3-4B NPUモデルを使うSemIfのフォークです。
直接判定とJSONの逐次生成を、同じモデルで比較できます。

初回セットアップ: docs/RYZENAI.md
モデルと依存関係を準備してから、リポジトリ内のPowerShellで実行:
  .\run_semif_npu_demo.ps1

  日本語版: http://127.0.0.1:8008/
  英語版:   http://127.0.0.1:8008/en/

上記URLは、このPCでサーバーを起動している間だけ利用できます。
8008が使用中の場合は .\run_semif_npu_demo.ps1 -Port 8009 と実行し、
両方のURLの8008を8009へ変更してください。
画面上の言語リンクでも切り替えられます。モデルは両ページで共用します。
初期表示の「16窓口」は、直接読出しと長いJSON出力の時間差を見る例です。
選択肢は最大32個です。32窓口の比較例も選択できます。
停止する場合はサーバーのターミナルでCtrl+Cを押してください。
JSONL入力を処理する場合は .\run_semif_npu.ps1 を実行します。

English
-------
This SemIf fork uses AMD Ryzen AI 1.8.0 and the official Qwen3-4B NPU model.
Compare direct option scoring with streamed JSON generation on the same model.

First-time setup: docs/RYZENAI.md
After installing the dependencies and downloading the model, run PowerShell
from the repository directory:
  .\run_semif_npu_demo.ps1

  Japanese demo: http://127.0.0.1:8008/
  English demo:  http://127.0.0.1:8008/en/

These URLs work while the local server is running on this PC. Use the language
links in the page to switch. Both pages share one loaded model.
If port 8008 is in use, run .\run_semif_npu_demo.ps1 -Port 8009 and replace
8008 with 8009 in both URLs.
The default 16-queue example highlights the time spent generating a full JSON
object. Up to 32 options are supported; a 32-queue comparison is also available.
Press Ctrl+C in the server terminal to stop it.
For JSONL scoring, run .\run_semif_npu.ps1.

Notes / 注意
------------
The official AMD NPU configuration includes CPU components; GPU offload is not
configured. Option scores and generated probabilities are not calibrated
confidence. SemIf is an independent project, not an official Jev binary or API.
AMD公式構成にはCPU処理も含まれます。確率表示は校正済みの信頼度ではありません。
