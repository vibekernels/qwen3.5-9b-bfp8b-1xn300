#include "inference.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <random>
#include <sstream>

// Suppress warnings from third-party headers under nvcc
#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20208
#pragma nv_diag_suppress 611

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

static std::mutex g_inference_mutex;

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
    auto& tokenizer = get_tokenizer();
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

                {
                    std::lock_guard<std::mutex> lock(g_inference_mutex);
                    generate(prompt_tokens, max_tokens, temperature,
                        [&](int token_id, const std::string& text) -> bool {
                            json chunk = {
                                {"id", completion_id},
                                {"object", "chat.completion.chunk"},
                                {"created", created},
                                {"model", model_name},
                                {"choices", json::array({
                                    {{"index", 0}, {"delta", {{"content", text}}}, {"finish_reason", nullptr}}
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
            total_tokens = generate(prompt_tokens, max_tokens, temperature,
                [&](int token_id, const std::string& text) -> bool {
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
</style>
<script src="https://cdn.jsdelivr.net/npm/marked@15/marked.min.js"></script>
</head>
<body>
<header>
  <h1>Qwen3.5-9B</h1>
  <span class="model">BF16 &bull; CUDA</span>
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
const prompt = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const clearBtn = document.getElementById('clear-btn');
let messages = [];
let generating = false;

function autoResize() {
  prompt.style.height = 'auto';
  prompt.style.height = Math.min(prompt.scrollHeight, 200) + 'px';
}
prompt.addEventListener('input', autoResize);

marked.setOptions({ breaks: true, gfm: true });

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
  const text = prompt.value.trim();
  if (!text || generating) return;

  generating = true;
  sendBtn.disabled = true;
  prompt.value = '';
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
  prompt.focus();
}

sendBtn.addEventListener('click', send);
prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
clearBtn.addEventListener('click', () => {
  messages = [];
  chat.innerHTML = '';
  prompt.focus();
});
</script>
</body>
</html>
)html";

int main(int argc, char** argv) {
    std::string model_path;
    std::string host = "0.0.0.0";
    int port = 8080;
    int ctx_size = 262144;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
            ctx_size = atoi(argv[++i]);
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s -m <model_path> [--host <addr>] [--port <port>] [--ctx-size <n>]\n", argv[0]);
        return 1;
    }

    printf("Loading model: %s\n", model_path.c_str());
    if (!load_model_and_tokenizer(model_path.c_str(), ctx_size)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded. Context size: %d\n", ctx_size);

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

    printf("Server listening on %s:%d\n", host.c_str(), port);
    printf("  GET  /                    Chat UI\n");
    printf("  POST /v1/chat/completions\n");
    printf("  GET  /v1/models\n");
    printf("  GET  /health\n");

    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "Failed to bind to %s:%d\n", host.c_str(), port);
        return 1;
    }

    return 0;
}
