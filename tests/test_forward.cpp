// SPDX-License-Identifier: Apache-2.0
// Test: load model and run generation with a real prompt.

#include "engine.h"
#include "tokenizer.h"
#include <cstdio>
#include <vector>
#include <string>

// Format user message with Qwen ChatML template (thinking disabled)
static std::string apply_chat_template(const char* user_msg) {
    std::string result;
    result += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    result += "<|im_start|>user\n";
    result += user_msg;
    result += "<|im_end|>\n";
    result += "<|im_start|>assistant\n<think>\n\n</think>\n";
    return result;
}

int main(int argc, char** argv) {
    const char* model_path = (argc >= 2) ? argv[1] : "models/Qwen3.5-9B-BF16.gguf";
    const char* prompt = (argc >= 3) ? argv[2] : "What is the capital of France?";
    int max_tokens = (argc >= 4) ? atoi(argv[3]) : 128;
    bool raw_mode = (argc >= 5 && std::string(argv[4]) == "--raw");

    printf("=== Test Forward Pass ===\n");
    printf("Prompt: %s\n", prompt);
    printf("Max tokens: %d\n", max_tokens);

    if (!load_model_and_tokenizer(model_path, 512)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Apply chat template unless --raw mode
    std::string formatted;
    if (raw_mode) {
        formatted = prompt;
    } else {
        formatted = apply_chat_template(prompt);
    }

    // Tokenize the prompt
    auto& tokenizer = get_tokenizer();
    auto prompt_tokens = tokenizer.encode(formatted);
    printf("Prompt tokens (%zu): ", prompt_tokens.size());
    for (int t : prompt_tokens) printf("%d ", t);
    printf("\n");

    printf("\n--- Generation ---\n");

    StopReason reason;
    int generated = generate(prompt_tokens, max_tokens, 0.0f,
        [](int token_id, const std::string& text) -> bool {
            printf("%s", text.c_str());
            fflush(stdout);
            return true;  // continue generating
        }, &reason);

    printf("\n--- End ---\n");
    printf("Generated %d tokens, stop reason: %d (%s)\n", generated, reason,
           reason == STOP_EOS ? "EOS" :
           reason == STOP_LENGTH ? "LENGTH" :
           reason == STOP_CALLBACK ? "CALLBACK" : "?");

    shutdown();
    return 0;
}
