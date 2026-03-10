// Interactive chat CLI for Qwen 3.5-9B on Tenstorrent N300.
// Usage: qwen-chat [model.gguf] [context_size]

#include "engine.h"
#include "tokenizer.h"
#include "download.h"
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

static volatile sig_atomic_t g_interrupted = 0;
static int g_real_stdout_fd = -1;

static void sigint_handler(int) {
    g_interrupted = 1;
}

// Redirect stdout to /dev/null so engine printf noise is hidden.
// The real stdout fd is saved for writing chat output directly.
static void suppress_stdout() {
    fflush(stdout);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    close(devnull);
}

static void restore_stdout() {
    fflush(stdout);
    dup2(g_real_stdout_fd, STDOUT_FILENO);
}

// Write directly to the real stdout, bypassing any redirections.
static void write_out(const char* s, size_t len) {
    while (len > 0) {
        ssize_t n = write(g_real_stdout_fd, s, len);
        if (n <= 0) break;
        s += n;
        len -= n;
    }
}

static void write_out(const std::string& s) {
    write_out(s.data(), s.size());
}

// Build the full ChatML prompt from conversation history.
static std::string build_prompt(const std::vector<std::pair<std::string, std::string>>& history,
                                const std::string& current_user_msg) {
    std::string prompt;
    prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";

    for (auto& [user, assistant] : history) {
        prompt += "<|im_start|>user\n";
        prompt += user;
        prompt += "<|im_end|>\n";
        prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        prompt += assistant;
        prompt += "<|im_end|>\n";
    }

    prompt += "<|im_start|>user\n";
    prompt += current_user_msg;
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    return prompt;
}

int main(int argc, char** argv) {
    std::string model_spec = "unsloth/Qwen3.5-9B-GGUF:BF16";
    int ctx_size = 2048;

    if (const char* p = getenv("MODEL_PATH")) model_spec = p;
    if (argc >= 2) model_spec = argv[1];
    if (argc >= 3) ctx_size = atoi(argv[2]);

    // Force quiet mode to suppress per-token debug output
    setenv("QUIET", "1", 1);

    // Save real stdout fd, then redirect stdout to /dev/null during engine init
    g_real_stdout_fd = dup(STDOUT_FILENO);

    // Resolve model (download from HuggingFace if needed)
    std::string model_path = resolve_model(model_spec);
    if (model_path.empty()) {
        write_out("Failed to resolve model: " + model_spec + "\n");
        return 1;
    }

    write_out("Loading model (ctx=" + std::to_string(ctx_size) + ")...\n");

    suppress_stdout();
    bool ok = load_model_and_tokenizer(model_path.c_str(), ctx_size);
    restore_stdout();

    if (!ok) {
        write_out("Failed to load model\n");
        return 1;
    }

    write_out("Ready. Type your message and press Enter. Ctrl-C to exit.\n\n");

    // Set up SIGINT handler
    struct sigaction sa = {};
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);

    auto& tokenizer = get_tokenizer();
    int eos_id = tokenizer.eos_token_id();

    // Maintain token IDs incrementally for prefix caching.
    // Instead of re-tokenizing the entire conversation each turn, we only
    // tokenize new text and append those IDs — avoiding BPE boundary issues.
    std::vector<int> all_ids;
    bool first_turn = true;

    while (true) {
        write_out("> ");

        g_interrupted = 0;
        char buf[4096];
        if (!fgets(buf, sizeof(buf), stdin)) {
            write_out("\n");
            break;
        }

        std::string user_input(buf);
        while (!user_input.empty() && (user_input.back() == '\n' || user_input.back() == '\r'))
            user_input.pop_back();

        if (user_input.empty()) continue;

        // Build token IDs incrementally
        if (first_turn) {
            // First turn: tokenize the full prompt
            std::string prompt = build_prompt({}, user_input);
            all_ids = tokenizer.encode(prompt);
            first_turn = false;
        } else {
            // Subsequent turns: tokenize only the new suffix and append
            // Previous all_ids ends after the last generated token (no EOS).
            // We need: <|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
            std::string suffix = "<|im_end|>\n<|im_start|>user\n"
                + user_input
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            auto suffix_ids = tokenizer.encode(suffix);
            all_ids.insert(all_ids.end(), suffix_ids.begin(), suffix_ids.end());
        }

        // Check if prompt fits in context
        if ((int)all_ids.size() >= ctx_size - 4) {
            write_out("[context full — clearing history]\n");
            reset_state();
            first_turn = true;
            // Re-build with just the current message
            std::string prompt = build_prompt({}, user_input);
            all_ids = tokenizer.encode(prompt);
            first_turn = false;
        }

        int max_gen = ctx_size - (int)all_ids.size();
        if (max_gen < 1) max_gen = 1;

        g_interrupted = 0;
        std::string response;
        StopReason reason;

        // Suppress engine stdout during generate (hides [prefill: ...] etc.)
        suppress_stdout();

        generate(all_ids, max_gen, 0.6f,
            [&](int token_id, const std::string& text) -> bool {
                if (g_interrupted) return false;
                if (token_id == eos_id) return true;  // don't print <|im_end|>
                write_out(text);
                response += text;
                all_ids.push_back(token_id);  // track for prefix caching
                return true;
            }, &reason);

        restore_stdout();

        write_out("\n\n");
    }

    suppress_stdout();
    shutdown();

    return 0;
}
