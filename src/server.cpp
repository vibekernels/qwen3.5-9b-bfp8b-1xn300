#include "engine.h"
#include "tokenizer.h"
#include "download.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <atomic>

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

static std::mutex g_inference_mutex;

// Server loading state
enum class ServerState { DOWNLOADING, LOADING, READY, FAILED };
static std::atomic<ServerState> g_state{ServerState::DOWNLOADING};
static DownloadProgress g_download_progress;
static std::string g_error_msg;

static std::string generate_id() {
    static std::mt19937 rng(std::random_device{}());
    static const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    std::string id = "chatcmpl-";
    for (int i = 0; i < 16; i++) {
        id += chars[rng() % (sizeof(chars) - 1)];
    }
    return id;
}

static int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Apply Qwen ChatML template to OpenAI-format messages
static std::string apply_chat_template(const json& messages) {
    std::string prompt;
    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");
        std::string content = msg.value("content", "");
        if (role == "assistant") {
            // Match the generation format so prompt cache can reuse prior turns
            prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + content + "<|im_end|>\n";
        } else {
            prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
    }
    // Add generation prompt with thinking disabled
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return prompt;
}

static void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    if (g_state.load() != ServerState::READY) {
        res.status = 503;
        res.set_content(R"({"error":{"message":"Model is still loading","type":"server_error"}})", "application/json");
        return;
    }

    json body;
    try {
        body = json::parse(req.body);
    } catch (...) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"Invalid JSON","type":"invalid_request_error"}})", "application/json");
        return;
    }

    auto messages = body.value("messages", json::array());
    if (messages.empty()) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"messages is required","type":"invalid_request_error"}})", "application/json");
        return;
    }

    int max_tokens = body.value("max_tokens", 1024);
    float temperature = body.value("temperature", 0.7f);
    bool stream = body.value("stream", false);
    std::string model_name = body.value("model", "qwen3.5-9b");

    // Apply chat template and tokenize
    std::string prompt_text = apply_chat_template(messages);
    const auto& tokenizer = get_tokenizer();
    std::vector<int> prompt_tokens = tokenizer.encode(prompt_text);

    std::string completion_id = generate_id();
    int64_t created = unix_timestamp();

    if (stream) {
        // SSE streaming response
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no");

        // Send initial role delta
        res.set_chunked_content_provider("text/event-stream; charset=utf-8",
            [&, prompt_tokens, max_tokens, temperature, model_name, completion_id, created]
            (size_t offset, httplib::DataSink& sink) -> bool {

                // Send role delta first
                json role_chunk = {
                    {"id", completion_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created},
                    {"model", model_name},
                    {"choices", json::array({
                        {{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}
                    })}
                };
                std::string role_data = "data: " + role_chunk.dump() + "\n\n";
                sink.write(role_data.c_str(), role_data.size());

                StopReason reason;
                std::string utf8_buf;  // buffer for incomplete UTF-8 sequences

                {
                    std::lock_guard<std::mutex> lock(g_inference_mutex);
                    reset_state();
                    generate(prompt_tokens, max_tokens, temperature,
                        [&](int token_id, const std::string& text) -> bool {
                            if (token_id == tokenizer.eos_token_id()) return true;
                            utf8_buf += text;
                            // Find last complete UTF-8 boundary
                            size_t complete = utf8_buf.size();
                            if (complete > 0) {
                                // Walk back from end to find incomplete sequence
                                size_t i = complete;
                                while (i > 0 && (utf8_buf[i-1] & 0xC0) == 0x80) i--;  // skip continuation bytes
                                if (i > 0) {
                                    unsigned char lead = utf8_buf[i-1];
                                    int expected = 1;
                                    if ((lead & 0xE0) == 0xC0) expected = 2;
                                    else if ((lead & 0xF0) == 0xE0) expected = 3;
                                    else if ((lead & 0xF8) == 0xF0) expected = 4;
                                    if (complete - (i-1) < (size_t)expected)
                                        complete = i - 1;  // incomplete multi-byte char, don't send yet
                                }
                            }
                            if (complete == 0) return true;  // nothing complete to send yet
                            std::string to_send = utf8_buf.substr(0, complete);
                            utf8_buf.erase(0, complete);
                            json chunk = {
                                {"id", completion_id},
                                {"object", "chat.completion.chunk"},
                                {"created", created},
                                {"model", model_name},
                                {"choices", json::array({
                                    {{"index", 0}, {"delta", {{"content", to_send}}}, {"finish_reason", nullptr}}
                                })}
                            };
                            std::string data = "data: " + chunk.dump() + "\n\n";
                            return sink.write(data.c_str(), data.size());
                        }, &reason);
                }

                std::string finish_reason = (reason == STOP_EOS) ? "stop" : "length";

                // Send final chunk with finish_reason
                json final_chunk = {
                    {"id", completion_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created},
                    {"model", model_name},
                    {"choices", json::array({
                        {{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason}}
                    })}
                };
                std::string final_data = "data: " + final_chunk.dump() + "\n\n";
                sink.write(final_data.c_str(), final_data.size());

                // Send [DONE]
                std::string done = "data: [DONE]\n\n";
                sink.write(done.c_str(), done.size());
                sink.done();
                return true;
            }
        );
    } else {
        // Non-streaming response
        std::string full_response;
        int total_tokens = 0;
        StopReason reason;

        {
            std::lock_guard<std::mutex> lock(g_inference_mutex);
            reset_state();
            total_tokens = generate(prompt_tokens, max_tokens, temperature,
                [&](int token_id, const std::string& text) -> bool {
                    if (token_id == tokenizer.eos_token_id()) return true;
                    full_response += text;
                    return true;
                }, &reason);
        }

        std::string finish_reason = (reason == STOP_EOS) ? "stop" : "length";

        json response = {
            {"id", completion_id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", model_name},
            {"choices", json::array({
                {{"index", 0},
                 {"message", {{"role", "assistant"}, {"content", full_response}}},
                 {"finish_reason", finish_reason}}
            })},
            {"usage", {
                {"prompt_tokens", (int)prompt_tokens.size()},
                {"completion_tokens", total_tokens},
                {"total_tokens", (int)prompt_tokens.size() + total_tokens}
            }}
        };

        res.set_content(response.dump(), "application/json");
    }
}

static void handle_models(const httplib::Request&, httplib::Response& res) {
    json response = {
        {"object", "list"},
        {"data", json::array({
            {{"id", "qwen3.5-9b"}, {"object", "model"}, {"owned_by", "local"}}
        })}
    };
    res.set_content(response.dump(), "application/json");
}

static void handle_health(const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
}

static void handle_status(const httplib::Request&, httplib::Response& res) {
    json status;
    ServerState state = g_state.load();
    switch (state) {
        case ServerState::DOWNLOADING:
            status["state"] = "downloading";
            status["downloaded"] = g_download_progress.downloaded.load();
            status["total"] = g_download_progress.total.load();
            status["filename"] = g_download_progress.filename;
            break;
        case ServerState::LOADING:
            status["state"] = "loading";
            break;
        case ServerState::READY:
            status["state"] = "ready";
            break;
        case ServerState::FAILED:
            status["state"] = "failed";
            status["error"] = g_error_msg;
            break;
    }
    res.set_content(status.dump(), "application/json");
}

static const char* CHAT_HTML = R"html(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3.5-9B Chat</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; height: 100vh; display: flex; flex-direction: column; }
header { padding: 12px 20px; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
header h1 { font-size: 16px; font-weight: 600; }
header .model { font-size: 12px; color: #8b949e; background: #21262d; padding: 2px 8px; border-radius: 12px; }
#chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
.msg { max-width: 780px; width: 100%; margin: 0 auto; display: flex; gap: 12px; }
.msg.user { justify-content: flex-end; }
.msg .bubble { padding: 10px 16px; border-radius: 12px; line-height: 1.5; word-break: break-word; max-width: 85%; font-size: 14px; }
.msg.user .bubble { background: #1f6feb; color: #fff; border-bottom-right-radius: 4px; white-space: pre-wrap; }
.msg.assistant .bubble { background: #161b22; border: 1px solid #30363d; border-bottom-left-radius: 4px; }
.msg.assistant .bubble code { background: #0d1117; padding: 1px 5px; border-radius: 4px; font-size: 13px; font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; }
.msg.assistant .bubble pre { background: #0d1117; padding: 12px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
.msg.assistant .bubble pre code { background: none; padding: 0; }
.msg.assistant .bubble p { margin: 0 0 8px; }
.msg.assistant .bubble p:last-child { margin-bottom: 0; }
.msg.assistant .bubble ul, .msg.assistant .bubble ol { margin: 4px 0 8px 20px; }
.msg.assistant .bubble li { margin: 2px 0; }
.msg.assistant .bubble h1, .msg.assistant .bubble h2, .msg.assistant .bubble h3 { margin: 12px 0 6px; font-size: 15px; }
.msg.assistant .bubble h1 { font-size: 17px; }
.msg.assistant .bubble blockquote { border-left: 3px solid #30363d; padding-left: 12px; color: #8b949e; margin: 8px 0; }
.msg.assistant .bubble table { border-collapse: collapse; margin: 8px 0; }
.msg.assistant .bubble th, .msg.assistant .bubble td { border: 1px solid #30363d; padding: 4px 8px; font-size: 13px; }
.msg.assistant .bubble th { background: #161b22; }
.msg.assistant .bubble hr { border: none; border-top: 1px solid #30363d; margin: 12px 0; }
#input-area { padding: 12px 20px 20px; border-top: 1px solid #30363d; flex-shrink: 0; }
#input-row { max-width: 780px; margin: 0 auto; display: flex; gap: 8px; }
#prompt { flex: 1; background: #161b22; border: 1px solid #30363d; color: #e6edf3; padding: 10px 14px; border-radius: 8px; font-size: 14px; font-family: inherit; resize: none; outline: none; min-height: 44px; max-height: 200px; }
#prompt:focus { border-color: #1f6feb; }
#prompt::placeholder { color: #484f58; }
#send { background: #1f6feb; color: #fff; border: none; padding: 0 20px; border-radius: 8px; font-size: 14px; cursor: pointer; font-weight: 500; white-space: nowrap; }
#send:hover { background: #388bfd; }
#send:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
.typing { display: inline-block; }
.typing::after { content: '\25CF'; animation: blink 1s infinite; }
@keyframes blink { 0%, 100% { opacity: .2; } 50% { opacity: 1; } }
#settings { display: flex; gap: 12px; max-width: 780px; margin: 0 auto 8px; align-items: center; font-size: 12px; color: #8b949e; }
#settings label { display: flex; align-items: center; gap: 4px; }
#settings input, #settings select { background: #161b22; border: 1px solid #30363d; color: #e6edf3; padding: 2px 6px; border-radius: 4px; font-size: 12px; width: 70px; }
#loading-overlay { position: fixed; inset: 0; background: #0d1117; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 100; gap: 16px; }
#loading-overlay h2 { font-size: 18px; font-weight: 600; }
#loading-overlay .status-text { font-size: 14px; color: #8b949e; }
#loading-overlay .progress-bar { width: 320px; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; }
#loading-overlay .progress-fill { height: 100%; background: #1f6feb; border-radius: 4px; transition: width 0.3s; width: 0%; }
#loading-overlay .progress-detail { font-size: 12px; color: #484f58; }
#loading-overlay .error { color: #f85149; font-size: 14px; }
#loading-overlay .spinner { width: 24px; height: 24px; border: 3px solid #21262d; border-top-color: #1f6feb; border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
<script src="https://cdn.jsdelivr.net/npm/marked@15/marked.min.js"></script>
</head>
<body>
<div id="loading-overlay">
  <h2>Qwen3.5-9B</h2>
  <div class="spinner"></div>
  <div class="status-text" id="load-status">Connecting...</div>
  <div class="progress-bar" id="progress-bar" style="display:none"><div class="progress-fill" id="progress-fill"></div></div>
  <div class="progress-detail" id="progress-detail"></div>
  <div class="error" id="load-error" style="display:none"></div>
</div>
<header>
  <h1>Qwen3.5-9B</h1>
  <span class="model">BFP8B &bull; Tenstorrent N300</span>
</header>
<div id="chat"></div>
<div id="input-area">
  <div id="settings">
    <label>Temp <input type="number" id="temp" value="0.7" min="0" max="2" step="0.1"></label>
    <label>Max tokens <input type="number" id="max-tok" value="1024" min="1" max="4096" step="64"></label>
    <button id="clear-btn" style="background:none;border:1px solid #30363d;color:#8b949e;padding:2px 10px;border-radius:4px;cursor:pointer;font-size:12px;">Clear</button>
  </div>
  <div id="input-row">
    <textarea id="prompt" rows="1" placeholder="Send a message..." autofocus></textarea>
    <button id="send">Send</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const clearBtn = document.getElementById('clear-btn');
const overlay = document.getElementById('loading-overlay');
let messages = [];
let generating = false;

function autoResize() {
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + 'px';
}
promptEl.addEventListener('input', autoResize);

marked.setOptions({ breaks: true, gfm: true });

// Loading state management
function formatBytes(b) {
  if (b < 1e6) return (b / 1e3).toFixed(0) + ' KB';
  if (b < 1e9) return (b / 1e6).toFixed(1) + ' MB';
  return (b / 1e9).toFixed(2) + ' GB';
}

async function pollStatus() {
  const statusEl = document.getElementById('load-status');
  const progressBar = document.getElementById('progress-bar');
  const progressFill = document.getElementById('progress-fill');
  const progressDetail = document.getElementById('progress-detail');
  const errorEl = document.getElementById('load-error');
  const spinnerEl = overlay.querySelector('.spinner');

  while (true) {
    try {
      const res = await fetch('/api/status', { signal: AbortSignal.timeout(2000) });
      const s = await res.json();

      if (s.state === 'downloading') {
        progressBar.style.display = '';
        const pct = s.total > 0 ? (s.downloaded / s.total * 100) : 0;
        progressFill.style.width = pct.toFixed(1) + '%';
        statusEl.textContent = 'Downloading model...';
        if (s.total > 0) {
          progressDetail.textContent = formatBytes(s.downloaded) + ' / ' + formatBytes(s.total) + ' (' + pct.toFixed(1) + '%)';
        } else {
          progressDetail.textContent = formatBytes(s.downloaded);
        }
      } else if (s.state === 'loading') {
        statusEl.textContent = 'Loading model onto device...';
        progressBar.style.display = 'none';
        progressDetail.textContent = '';
      } else if (s.state === 'ready') {
        overlay.style.display = 'none';
        promptEl.focus();
        return;
      } else if (s.state === 'failed') {
        statusEl.textContent = 'Failed to load model';
        spinnerEl.style.display = 'none';
        progressBar.style.display = 'none';
        errorEl.style.display = '';
        errorEl.textContent = s.error || 'Unknown error';
        return;
      }
    } catch {
      statusEl.textContent = 'Connecting...';
    }
    await new Promise(r => setTimeout(r, 500));
  }
}
pollStatus();

function renderMarkdown(el, text) {
  try { el.innerHTML = marked.parse(text); } catch { el.textContent = text; }
}

function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (role === 'user') bubble.textContent = content;
  else renderMarkdown(bubble, content);
  div.appendChild(bubble);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

async function send() {
  const text = promptEl.value.trim();
  if (!text || generating) return;

  generating = true;
  sendBtn.disabled = true;
  promptEl.value = '';
  autoResize();

  messages.push({ role: 'user', content: text });
  addMessage('user', text);

  const bubble = addMessage('assistant', '');
  bubble.innerHTML = '<span class="typing"></span>';

  const temp = parseFloat(document.getElementById('temp').value) || 0.7;
  const maxTok = parseInt(document.getElementById('max-tok').value) || 1024;

  let fullText = '';
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'qwen3.5-9b', messages, stream: true, temperature: temp, max_tokens: maxTok })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
        try {
          const obj = JSON.parse(line.slice(6));
          const delta = obj.choices?.[0]?.delta;
          if (delta?.content) {
            fullText += delta.content;
            renderMarkdown(bubble, fullText);
            chat.scrollTop = chat.scrollHeight;
          }
        } catch {}
      }
    }
  } catch (e) {
    fullText = 'Error: ' + e.message;
    bubble.textContent = fullText;
  }

  messages.push({ role: 'assistant', content: fullText });
  generating = false;
  sendBtn.disabled = false;
  promptEl.focus();
}

sendBtn.addEventListener('click', send);
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
clearBtn.addEventListener('click', () => {
  messages = [];
  chat.innerHTML = '';
  promptEl.focus();
});
</script>
</body>
</html>
)html";

int main(int argc, char** argv) {
    std::string model_spec;
    std::string model_dir;
    std::string host = "0.0.0.0";
    int port = 8888;
    int ctx_size = 4096;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_spec = argv[++i];
        } else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
            ctx_size = atoi(argv[++i]);
        }
    }

    if (model_spec.empty()) {
        fprintf(stderr, "Usage: %s -m <model_or_hf_tag> [--model-dir <dir>] [--host <addr>] [--port <port>] [--ctx-size <n>]\n", argv[0]);
        fprintf(stderr, "\n  -m   Local .gguf file path, or HuggingFace tag like org/repo:quant\n");
        fprintf(stderr, "       Example: -m unsloth/Qwen3.5-9B-GGUF:BF16\n");
        fprintf(stderr, "  --model-dir  Local cache directory (default: ~/.cache/qwen-models)\n");
        return 1;
    }

    // Start model download + loading in a background thread
    std::thread loader([model_spec, model_dir, ctx_size]() {
        g_state.store(ServerState::DOWNLOADING);

        std::string model_path = resolve_model(model_spec, model_dir,
            [](int64_t downloaded, int64_t total) {
                g_download_progress.downloaded.store(downloaded);
                g_download_progress.total.store(total);
            });

        if (model_path.empty()) {
            g_error_msg = "Failed to resolve model: " + model_spec;
            g_state.store(ServerState::FAILED);
            return;
        }

        g_download_progress.filename = model_path;
        g_state.store(ServerState::LOADING);
        printf("Loading model: %s\n", model_path.c_str());

        if (!load_model_and_tokenizer(model_path.c_str(), ctx_size)) {
            g_error_msg = "Failed to load model";
            g_state.store(ServerState::FAILED);
            return;
        }

        printf("Model loaded. Context size: %d\n", ctx_size);
        g_state.store(ServerState::READY);
    });
    loader.detach();

    httplib::Server svr;

    // CORS headers for browser clients
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(CHAT_HTML, "text/html; charset=utf-8");
    });
    svr.Post("/v1/chat/completions", handle_chat_completions);
    svr.Get("/v1/models", handle_models);
    svr.Get("/health", handle_health);
    svr.Get("/api/status", handle_status);

    printf("Server listening on %s:%d\n", host.c_str(), port);
    printf("  GET  /                    Chat UI\n");
    printf("  POST /v1/chat/completions\n");
    printf("  GET  /v1/models\n");
    printf("  GET  /api/status\n");
    printf("  GET  /health\n");

    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "Failed to bind to %s:%d\n", host.c_str(), port);
        return 1;
    }

    return 0;
}
