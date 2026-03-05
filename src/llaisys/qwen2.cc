#include "llaisys/models/qwen2.h"
#include "../ops/ops.hpp"
#include "../utils.hpp"
#include "llaisys_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace llaisys {

struct Qwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;

    // Weights
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;

    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    size_t cache_pos;

    Qwen2Model(const LlaisysQwen2Meta &m, llaisysDeviceType_t dev)
        : meta(m), device(dev), cache_pos(0) {
        // Create embedding and output tensors
        in_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device);
        out_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device);
        out_norm_w = Tensor::create({meta.hs}, meta.dtype, device);

        // Initialize weight vectors
        attn_norm_w.resize(meta.nlayer);
        attn_q_w.resize(meta.nlayer);
        attn_q_b.resize(meta.nlayer);
        attn_k_w.resize(meta.nlayer);
        attn_k_b.resize(meta.nlayer);
        attn_v_w.resize(meta.nlayer);
        attn_v_b.resize(meta.nlayer);
        attn_o_w.resize(meta.nlayer);
        mlp_norm_w.resize(meta.nlayer);
        mlp_gate_w.resize(meta.nlayer);
        mlp_up_w.resize(meta.nlayer);
        mlp_down_w.resize(meta.nlayer);

        // Create per-layer weight tensors
        for (size_t i = 0; i < meta.nlayer; i++) {
            attn_norm_w[i] = Tensor::create({meta.hs}, meta.dtype, device);
            attn_q_w[i] = Tensor::create({meta.nh * meta.dh, meta.hs}, meta.dtype, device);
            attn_q_b[i] = Tensor::create({meta.nh * meta.dh}, meta.dtype, device);
            attn_k_w[i] = Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device);
            attn_k_b[i] = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device);
            attn_v_w[i] = Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device);
            attn_v_b[i] = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device);
            attn_o_w[i] = Tensor::create({meta.hs, meta.hs}, meta.dtype, device);
            mlp_norm_w[i] = Tensor::create({meta.hs}, meta.dtype, device);
            mlp_gate_w[i] = Tensor::create({meta.di, meta.hs}, meta.dtype, device);
            mlp_up_w[i] = Tensor::create({meta.di, meta.hs}, meta.dtype, device);
            mlp_down_w[i] = Tensor::create({meta.hs, meta.di}, meta.dtype, device);
        }

        // Initialize KV cache
        k_cache.resize(meta.nlayer);
        v_cache.resize(meta.nlayer);
        for (size_t i = 0; i < meta.nlayer; i++) {
            k_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device);
            v_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device);
        }
    }

    int64_t infer(int64_t *token_ids, size_t ntoken) {
        size_t seq_len = ntoken;

        // Create position ids
        auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device);
        std::vector<int64_t> pos_data(seq_len);
        for (size_t i = 0; i < seq_len; i++) {
            pos_data[i] = cache_pos + i;
        }
        pos_ids->load(pos_data.data());

        // Embedding lookup
        auto token_tensor = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device);
        token_tensor->load(token_ids);
        auto hidden = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
        ops::embedding(hidden, token_tensor, in_embed);

        // Process each layer
        for (size_t layer = 0; layer < meta.nlayer; layer++) {
            // Attention norm
            auto normed = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
            ops::rms_norm(normed, hidden, attn_norm_w[layer], meta.epsilon);

            // Q, K, V projections
            auto q = Tensor::create({seq_len, meta.nh * meta.dh}, meta.dtype, device);
            auto k = Tensor::create({seq_len, meta.nkvh * meta.dh}, meta.dtype, device);
            auto v = Tensor::create({seq_len, meta.nkvh * meta.dh}, meta.dtype, device);

            ops::linear(q, normed, attn_q_w[layer], attn_q_b[layer]);
            ops::linear(k, normed, attn_k_w[layer], attn_k_b[layer]);
            ops::linear(v, normed, attn_v_w[layer], attn_v_b[layer]);

            // Reshape to [seq_len, n_heads, head_dim]
            auto q_shaped = q->view({seq_len, meta.nh, meta.dh});
            auto k_shaped = k->view({seq_len, meta.nkvh, meta.dh});
            auto v_shaped = v->view({seq_len, meta.nkvh, meta.dh});

            // Apply RoPE
            auto q_rope = Tensor::create({seq_len, meta.nh, meta.dh}, meta.dtype, device);
            auto k_rope = Tensor::create({seq_len, meta.nkvh, meta.dh}, meta.dtype, device);
            ops::rope(q_rope, q_shaped, pos_ids, meta.theta);
            ops::rope(k_rope, k_shaped, pos_ids, meta.theta);

            // Update KV cache
            for (size_t i = 0; i < seq_len; i++) {
                auto k_slice = k_rope->slice(0, i, i + 1);
                auto v_slice = v_shaped->slice(0, i, i + 1);
                auto k_cache_slice = k_cache[layer]->slice(0, cache_pos + i, cache_pos + i + 1);
                auto v_cache_slice = v_cache[layer]->slice(0, cache_pos + i, cache_pos + i + 1);

                // Flatten to 1D and perform D2D copy
                auto k_flat = k_slice->view({meta.nkvh * meta.dh});
                auto v_flat = v_slice->view({meta.nkvh * meta.dh});
                auto kc_flat = k_cache_slice->view({meta.nkvh * meta.dh});
                auto vc_flat = v_cache_slice->view({meta.nkvh * meta.dh});

                size_t copy_bytes = meta.nkvh * meta.dh * k_flat->elementSize();
                core::context().runtime().api()->memcpy_sync(
                    kc_flat->data(), k_flat->data(), copy_bytes, LLAISYS_MEMCPY_D2D);
                core::context().runtime().api()->memcpy_sync(
                    vc_flat->data(), v_flat->data(), copy_bytes, LLAISYS_MEMCPY_D2D);
            }

            // Get full KV from cache
            size_t total_len = cache_pos + seq_len;
            auto k_full = k_cache[layer]->slice(0, 0, total_len);
            auto v_full = v_cache[layer]->slice(0, 0, total_len);

            // Self attention
            auto attn_out = Tensor::create({seq_len, meta.nh, meta.dh}, meta.dtype, device);
            float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
            ops::self_attention(attn_out, q_rope, k_full, v_full, scale);

            // Output projection
            auto attn_flat = attn_out->view({seq_len, meta.nh * meta.dh});
            auto attn_proj = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
            ops::linear(attn_proj, attn_flat, attn_o_w[layer], nullptr);

            // Residual connection
            ops::add(hidden, hidden, attn_proj);

            // MLP norm
            auto mlp_normed = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
            ops::rms_norm(mlp_normed, hidden, mlp_norm_w[layer], meta.epsilon);

            // MLP
            auto gate = Tensor::create({seq_len, meta.di}, meta.dtype, device);
            auto up = Tensor::create({seq_len, meta.di}, meta.dtype, device);
            ops::linear(gate, mlp_normed, mlp_gate_w[layer], nullptr);
            ops::linear(up, mlp_normed, mlp_up_w[layer], nullptr);

            auto mlp_out = Tensor::create({seq_len, meta.di}, meta.dtype, device);
            ops::swiglu(mlp_out, gate, up);

            auto mlp_proj = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
            ops::linear(mlp_proj, mlp_out, mlp_down_w[layer], nullptr);

            // Residual connection
            ops::add(hidden, hidden, mlp_proj);
        }

        // Final norm
        auto final_normed = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
        ops::rms_norm(final_normed, hidden, out_norm_w, meta.epsilon);

        // Get last token
        auto last_hidden = final_normed->slice(0, seq_len - 1, seq_len);
        auto last_flat = last_hidden->view({meta.hs});

        // LM head
        auto logits = Tensor::create({1, meta.voc}, meta.dtype, device);
        ops::linear(logits, last_flat->view({1, meta.hs}), out_embed, nullptr);

        // Argmax — max_val is always F32 because the CUDA kernel writes float
        auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device);
        auto max_val = Tensor::create({1}, LLAISYS_DTYPE_F32, device);
        auto logits_flat = logits->view({meta.voc});
        ops::argmax(max_idx, max_val, logits_flat);

        // Get result
        int64_t result;
        std::vector<int64_t> result_vec(1);
        core::context().runtime().api()->memcpy_sync(
            reinterpret_cast<std::byte *>(result_vec.data()),
            max_idx->data(),
            sizeof(int64_t),
            LLAISYS_MEMCPY_D2H);
        result = result_vec[0];

        // Update cache position
        cache_pos += seq_len;

        return result;
    }

    void reset_cache() {
        cache_pos = 0;
    }
};

} // namespace llaisys

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    auto model = new llaisys::Qwen2Model(*meta, device);
    return reinterpret_cast<LlaisysQwen2Model *>(model);
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete reinterpret_cast<llaisys::Qwen2Model *>(model);
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model_) {
    auto model = reinterpret_cast<llaisys::Qwen2Model *>(model_);
    auto weights = new LlaisysQwen2Weights();

    weights->in_embed = new LlaisysTensor{model->in_embed};
    weights->out_embed = new LlaisysTensor{model->out_embed};
    weights->out_norm_w = new LlaisysTensor{model->out_norm_w};

    weights->attn_norm_w = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_q_w = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_q_b = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_k_w = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_k_b = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_v_w = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_v_b = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_o_w = new llaisysTensor_t[model->meta.nlayer];
    weights->mlp_norm_w = new llaisysTensor_t[model->meta.nlayer];
    weights->mlp_gate_w = new llaisysTensor_t[model->meta.nlayer];
    weights->mlp_up_w = new llaisysTensor_t[model->meta.nlayer];
    weights->mlp_down_w = new llaisysTensor_t[model->meta.nlayer];

    for (size_t i = 0; i < model->meta.nlayer; i++) {
        weights->attn_norm_w[i] = new LlaisysTensor{model->attn_norm_w[i]};
        weights->attn_q_w[i] = new LlaisysTensor{model->attn_q_w[i]};
        weights->attn_q_b[i] = new LlaisysTensor{model->attn_q_b[i]};
        weights->attn_k_w[i] = new LlaisysTensor{model->attn_k_w[i]};
        weights->attn_k_b[i] = new LlaisysTensor{model->attn_k_b[i]};
        weights->attn_v_w[i] = new LlaisysTensor{model->attn_v_w[i]};
        weights->attn_v_b[i] = new LlaisysTensor{model->attn_v_b[i]};
        weights->attn_o_w[i] = new LlaisysTensor{model->attn_o_w[i]};
        weights->mlp_norm_w[i] = new LlaisysTensor{model->mlp_norm_w[i]};
        weights->mlp_gate_w[i] = new LlaisysTensor{model->mlp_gate_w[i]};
        weights->mlp_up_w[i] = new LlaisysTensor{model->mlp_up_w[i]};
        weights->mlp_down_w[i] = new LlaisysTensor{model->mlp_down_w[i]};
    }

    return weights;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model_, int64_t *token_ids, size_t ntoken) {
    auto model = reinterpret_cast<llaisys::Qwen2Model *>(model_);
    return model->infer(token_ids, ntoken);
}

} // extern "C"
