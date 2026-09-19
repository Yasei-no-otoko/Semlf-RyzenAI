const $ = (selector) => document.querySelector(selector);
const ENGLISH = document.documentElement.lang === "en";
const tr = (japanese, english) => ENGLISH ? english : japanese;
const MIN_OPTIONS = 2;
const MAX_OPTIONS = 32;
const OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345";
const MODEL = "AMD Qwen3-4B Ryzen AI 1.8 NPU 4K";
const REVISION = "d6fb03663d78ae5034d4594bfe9d92b35a5e213a";
const japanesePresets = {
  account: {
    state: "パスワードリセットは成功したが、その後もすべてのログインが「アカウントロック」になる。解除メールを2回依頼したが届いていない。",
    question: "この依頼はどのキューで処理すべきですか？",
    options: ["アカウントアクセス", "請求", "解決済みとして終了"],
  },
  email: {
    state: "給与チームを名乗り、今日中に支払いを停止すると伝えるメールが届いた。送信元は payroll-review@outlook.com で、社外のサインインページへのリンクがあり、パスワードと確認コードを求めている。",
    question: "このメールをどう分類すべきですか？",
    options: ["正当なメール", "スパム", "フィッシング"],
  },
  routing16: {
    state: "お客様からの依頼：先月購入した商品の領収書を、会社の経費精算に使うためPDFで再発行してください。購入と支払いは完了しています。返金や返品は希望していません。ログインも問題ありません。",
    question: "問い合わせの主目的に最も合う窓口を1つ選んでください。",
    options: [
      "パスワードの再設定", "二段階認証の復旧", "メールアドレスの変更", "アカウントの削除",
      "請求内容の照会", "領収書の発行", "返金の申請", "支払い方法の変更",
      "配送状況の確認", "配送先の変更", "商品の返品", "商品の交換",
      "製品の使い方", "不具合の報告", "法人契約の相談", "その他の問い合わせ",
    ],
  },
  routing32: {
    state: "お客様からの依頼：先月購入した商品の領収書を、会社の経費精算に使うためPDFで再発行してください。購入と支払いは完了しています。返金や返品は希望していません。ログインも問題ありません。",
    question: "問い合わせの主目的に最も合う窓口を1つ選んでください。",
    options: ["パスワードの再設定", "二段階認証の復旧", "メールアドレスの変更", "アカウントの削除", "請求内容の照会", "領収書の発行", "返金の申請", "支払い方法の変更", "配送状況の確認", "配送先の変更", "商品の返品", "商品の交換", "製品の使い方", "不具合の報告", "法人契約の相談", "その他の問い合わせ", "注文履歴の確認", "ギフトカードの残高", "クーポンの利用", "定期購入の変更", "アカウント名義の変更", "プライバシー設定", "通知設定", "保証の延長", "在庫の確認", "納期の問い合わせ", "配送業者の変更", "受取日時の変更", "請求先住所の変更", "税務書類の依頼", "領収書の宛名変更", "購入証明の依頼"],
  },
};
const englishPresets = {
  account: {
    state: "A password reset succeeded, but every login still returns ‘account locked’. Two unlock emails were requested and neither arrived.",
    question: "Which queue should handle this request?",
    options: ["Account access support", "Billing support", "Close as resolved"],
  },
  email: {
    state: "An email claims to be from the payroll team and says payment will be suspended today. It comes from payroll-review@outlook.com and links to an external sign-in page asking for a password and verification code.",
    question: "How should this email be classified?",
    options: ["Legitimate", "Spam", "Phishing"],
  },
  routing16: {
    state: "A customer asks for a PDF receipt for a completed purchase so it can be submitted for company expenses. They do not want a refund or return and can sign in normally.",
    question: "Choose the single support route that best matches the request.",
    options: ["Reset password", "Recover two-factor authentication", "Change email address", "Delete account", "Ask about billing", "Issue a receipt", "Request a refund", "Change payment method", "Check delivery status", "Change delivery address", "Return product", "Exchange product", "Product how-to", "Report a defect", "Business contract help", "Other inquiry"],
  },
  routing32: {
    state: "A customer asks for a PDF receipt for a completed purchase so it can be submitted for company expenses. They do not want a refund or return and can sign in normally.",
    question: "Choose the single support route that best matches the request.",
    options: ["Reset password", "Recover two-factor authentication", "Change email address", "Delete account", "Ask about billing", "Issue a receipt", "Request a refund", "Change payment method", "Check delivery status", "Change delivery address", "Return product", "Exchange product", "Product how-to", "Report a defect", "Business contract help", "Other inquiry", "Check order history", "Gift card balance", "Use a coupon", "Change subscription", "Change account name", "Privacy settings", "Notification settings", "Extend warranty", "Check stock", "Ask about lead time", "Change carrier", "Change delivery time", "Change billing address", "Request tax document", "Change receipt name", "Request proof of purchase"],
  },
};
const presets = ENGLISH ? englishPresets : japanesePresets;
let busy = false;
let currentStatus = "unloaded";
let statusPoll = null;
let directSeconds = null;

function setText(selector, value) { $(selector).textContent = String(value); }
function setStatus(text, kind = "") {
  const node = $("#status");
  node.textContent = text;
  node.className = `support ${kind ? `status-${kind}` : ""}`.trim();
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll("button, input, textarea").forEach((node) => { node.disabled = value; });
  $("#run").disabled = value || currentStatus !== "ready";
  $("#load").disabled = value || currentStatus === "ready" || currentStatus === "loading" || currentStatus === "running";
  updateOptionButtons();
}
function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined || seconds === "") return "—";
  const value = Number(seconds);
  return Number.isFinite(value) ? `${value.toFixed(3)} ${tr("秒", "s")}` : "—";
}
function renderOptions(options) {
  const list = $("#option-list");
  list.replaceChildren();
  options.forEach((value, index) => {
    const label = document.createElement("label"); label.className = "option-row";
    const letter = document.createElement("b"); letter.textContent = OPTION_LABELS[index];
    const input = document.createElement("input"); input.className = "option"; input.value = value; input.type = "text";
    label.append(letter, input); list.append(label);
  });
  setText("#option-count", `${options.length} / ${MAX_OPTIONS}`);
  updateOptionButtons();
}
function updateOptionButtons() {
  const count = document.querySelectorAll(".option").length;
  $("#remove-option").disabled = count <= MIN_OPTIONS || busy;
  $("#add-option").disabled = count >= MAX_OPTIONS || busy;
}
function readOptions() { return [...document.querySelectorAll(".option")].map((node) => node.value.trim()); }
function applyPreset(name) {
  const preset = presets[name]; if (!preset || busy) return;
  $("#state").value = preset.state; $("#question").value = preset.question; renderOptions(preset.options);
}
async function jsonResponse(response) {
  let body = null; try { body = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) throw new Error(body?.error || `HTTP ${response.status}`);
  return body || {};
}
function scheduleStatusPoll() {
  if (statusPoll !== null) clearTimeout(statusPoll);
  statusPoll = null;
  if (busy || !["loading", "running"].includes(currentStatus)) return;
  statusPoll = setTimeout(() => { statusPoll = null; if (!busy) refreshStatus(); }, 1500);
}
async function refreshStatus(operationError = "") {
  try {
    const data = await jsonResponse(await fetch("/api/status", { headers: { Accept: "application/json" } }));
    currentStatus = data.state || "error";
    setText("#model-state", currentStatus);
    if (data.model) {
      setText("#model-name", data.model.name || MODEL);
      setText("#model-meta", `${data.model.backend || "Ryzen AI NPU"} · revision ${data.model.revision || REVISION} · ${tr("ローカル配置済み", "local model")}`);
      setText("#context-limit", data.model.context_limit ? `${data.model.context_limit} tok` : "—");
    }
    if (data.load_seconds != null) setText("#load-time", formatSeconds(data.load_seconds));
    if (currentStatus === "ready") { setStatus(tr("モデル準備完了。実行できます。", "Model ready. You can run a decision."), "ok"); }
    else if (currentStatus === "loading") setStatus(tr("モデルを読み込んでいます…", "Loading the model…"), "running");
    else if (currentStatus === "running") setStatus(tr("推論を実行中です…", "Inference is running…"), "running");
    else if (currentStatus === "error") setStatus(data.error || tr("サーバーでエラーが発生しました。", "The server reported an error."), "error");
    else setStatus(tr("モデルは未読み込みです。", "The model is not loaded."), "");
    if (operationError) setStatus(operationError, "error");
    setBusy(busy);
  } catch (error) { currentStatus = "error"; setText("#model-state", "error"); setStatus(operationError || `${tr("状態取得に失敗", "Status request failed")}: ${error.message}`, "error"); setBusy(busy); }
  scheduleStatusPoll();
}
async function loadModel() {
  if (busy || currentStatus === "ready") return;
  let operationError = "";
  setBusy(true); $("#load").classList.add("loading"); setStatus(tr("モデルを一度だけ読み込んでいます…", "Loading the model once…"), "running"); setText("#model-state", "loading");
  try { const data = await jsonResponse(await fetch("/api/load", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })); currentStatus = data.state || "ready"; setText("#load-time", formatSeconds(data.load_seconds)); if (data.model?.context_limit) setText("#context-limit", `${data.model.context_limit} tok`); setStatus(tr("モデル準備完了。実行できます。", "Model ready. You can run a decision."), "ok"); }
  catch (error) { currentStatus = "error"; operationError = `${tr("読み込みに失敗", "Model load failed")}: ${error.message}`; setStatus(operationError, "error"); }
  $("#load").classList.remove("loading"); setBusy(false); await refreshStatus(operationError);
}
function resetResults(running = false) {
  $("#direct-output").textContent = tr("直接読出しを待っています…", "Waiting for direct readout…"); $("#direct-output").className = "output empty";
  $("#generated-output").textContent = tr("JSON生成を待っています…", "Waiting for JSON generation…"); $("#generated-output").className = "output empty";
  $("#generation-validation").textContent = "—"; $("#generation-validation").className = "validation";
  directSeconds = null; ["#direct-total", "#direct-input", "#direct-readouts", "#generation-ttft", "#generation-total", "#generation-input", "#generation-tokens"].forEach((id) => setText(id, "—")); setText("#ratio", running ? tr("測定中…", "Measuring…") : tr("実行してください", "Run a decision"));
}
function renderDirect(result, options) {
  const output = $("#direct-output"); output.replaceChildren(); output.className = "output";
  const probabilities = result.probabilities || [];
  options.forEach((description, index) => {
    const row = document.createElement("div"); row.className = "choice";
    const label = document.createElement("span"); label.className = "choice-label";
    const letter = document.createElement("b"); letter.textContent = OPTION_LABELS[index];
    const text = document.createElement("small"); text.textContent = description; label.append(letter, text);
    const bar = document.createElement("span"); bar.className = "bar"; const fill = document.createElement("i"); fill.style.width = `${Math.max(1, Number(probabilities[index] || 0) * 100)}%`; bar.append(fill);
    const score = document.createElement("em"); score.textContent = Number(probabilities[index] || 0).toFixed(3); row.append(label, bar, score); output.append(row);
  });
  directSeconds = Number.isFinite(Number(result.total_seconds)) ? Number(result.total_seconds) : null;
  setText("#direct-total", formatSeconds(result.total_seconds)); setText("#direct-input", `${result.input_tokens ?? "—"} tok`); setText("#direct-readouts", "1 readout");
}
function renderGeneration(result) {
  if (typeof result.text === "string") { $("#generated-output").textContent = result.text; $("#generated-output").className = "output"; }
  setText("#generation-ttft", formatSeconds(result.ttft_seconds)); setText("#generation-total", formatSeconds(result.total_seconds)); setText("#generation-input", `${result.input_tokens ?? "—"} tok`); setText("#generation-tokens", result.output_tokens ?? "—");
  const node = $("#generation-validation"); const valid = result.valid_json && !result.truncated; node.textContent = valid ? tr("JSON検証済み", "Valid JSON") : (result.validation_error || tr("JSONを検証できませんでした", "JSON validation failed")); node.className = `validation ${valid ? "ok" : "error"}`;
  const generated = Number(result.total_seconds); if (directSeconds > 0 && generated > 0) setText("#ratio", `${(generated / directSeconds).toFixed(2)}×`);
}
async function run() {
  if (busy || currentStatus !== "ready") return;
  const state = $("#state").value.trim(); const question = $("#question").value.trim(); const options = readOptions();
  if (!state || !question || options.some((item) => !item) || options.length < MIN_OPTIONS || options.length > MAX_OPTIONS) { setStatus(tr("状態・質問・2〜32個の選択肢を入力してください。", "Enter state, question, and 2–32 non-empty options."), "error"); return; }
  let operationError = "";
  resetResults(true); setBusy(true); setStatus(tr("直接読出しとJSON生成を実行中です…", "Running direct readout and JSON generation…"), "running");
  try {
    const response = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" }, body: JSON.stringify({ state, question, options }) });
    if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.error || `HTTP ${response.status}`); }
    if (!response.body) throw new Error(tr("ストリームが利用できません。", "The response stream is unavailable."));
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let generationText = ""; let sawDirect = false; let sawGeneration = false; let sawDone = false;
    const consume = (line) => { if (!line.trim()) return; const event = JSON.parse(line); if (event.type === "direct") { sawDirect = true; renderDirect(event.result || {}, options); } else if (event.type === "token") { generationText += event.text || ""; $("#generated-output").textContent = generationText; $("#generated-output").className = "output"; } else if (event.type === "generation") { sawGeneration = true; renderGeneration(event.result || {}); } else if (event.type === "done") sawDone = true; else if (event.type === "error") throw new Error(event.error || tr("推論エラー", "Inference error")); };
    while (true) { const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true }); const lines = buffer.split("\n"); buffer = lines.pop(); lines.forEach(consume); }
    buffer += decoder.decode(); consume(buffer);
    if (!sawDone) throw new Error(tr("ストリームが完了イベントなしで切断されました。", "The stream disconnected before its done event."));
    if (!sawDirect || !sawGeneration) throw new Error(tr("直接読出しまたはJSON生成の完了結果がありません。", "The direct or generation result is missing."));
    setStatus(tr("実行完了。時間はこの実行の測定値です。", "Complete. Times are from this run."), "ok");
  } catch (error) { operationError = `${tr("実行に失敗", "Run failed")}: ${error.message}`; setStatus(operationError, "error"); setText("#ratio", tr("実行失敗", "Run failed")); }
  setBusy(false); await refreshStatus(operationError);
}
$("#load").addEventListener("click", loadModel); $("#run").addEventListener("click", run);
$("#add-option").addEventListener("click", () => { const options = readOptions(); if (options.length < MAX_OPTIONS) { renderOptions([...options, ""]); resetResults(); } });
$("#remove-option").addEventListener("click", () => { const options = readOptions(); if (options.length > MIN_OPTIONS) { renderOptions(options.slice(0, -1)); resetResults(); } });
document.querySelectorAll("[data-preset]").forEach((button) => button.addEventListener("click", () => { applyPreset(button.dataset.preset); resetResults(); }));
$(".inputs").addEventListener("input", () => { if (!busy) resetResults(); });
applyPreset("routing16"); refreshStatus();
