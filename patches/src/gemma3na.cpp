#include "models.h"

// Gemma 3n Audio Conformer graph builder
// Architecture: Subsample Conv -> Conformer layers -> Audio Embedding Projection
//
// Tensor format: [features, time] throughout (column-major for ggml_mul_mat)
// Input: mel spectrogram [n_frames, n_mel] = [T, 128]
// Conv subsampling reduces to [1024, T/4]
// Input projection: [1024, T/4] -> [1536, T/4]
// 12 Conformer blocks
// Audio embedding projection: [1536, T/4] -> [2048, T/4], then temporal reduction -> ~188 tokens

// Relative position encoding constants
static constexpr int G3NA_MAX_SPAN = 13;  // max_backward + 1 = 12 + 1
static constexpr int G3NA_N_CHANNELS = 1536;
static constexpr int G3NA_AUDIO_SOFT_TOKENS = 188;
static constexpr int G3NA_AUDIO_TOKENS_OUT = G3NA_AUDIO_SOFT_TOKENS; // only the 188 soft tokens; <end_of_audio> is a normal vocab token

// NOTE: Gemma3n uses CumulativeGroupNorm in the conv blocks which computes
// cumulative mean and variance over the time dimension. This is different from
// a simple per-position normalization. The CumulativeGroupNorm algorithm:
// 1. cum_mean[t] = sum(x[0:t+1]) / count[0:t+1]
// 2. cum_var[t] = sum((x[0:t+1] - cum_mean[t])^2) / count[0:t+1]
// 3. normalized = (x - cum_mean) / sqrt(cum_var + eps) * scale
//
// Current implementation uses layer norm (mean + variance) as an approximation
// since it computes mean/variance like CumulativeGroupNorm, but over all positions
// rather than cumulatively. This is closer than RMS norm but not exact.
// TODO: Implement true CumulativeGroupNorm with cumulative sum operations for
// potentially better audio transcription accuracy.

// Gemma3nAudioCumulativeGroupNorm (HF reference):
// - Input:  x [B, T, F, C] (after conv output is permuted)
// - Reduce: sum over (F, C), then cumulative mean/variance over T
// - Output: normalized_x * weight[C] (no bias), returned in original dtype
//
// In this graph we keep conv tensors as [T, F, C, B] (time-major). This helper
// implements the same math for the common B=1 case.
static ggml_tensor * gemma3na_cumulative_group_norm(
        ggml_context * ctx0,
        ggml_tensor  * x_ftc1,   // [F, T, C, 1] (freq, time, channel, batch)
        ggml_tensor  * w_c,      // [C]
        const float    eps,
        ggml_tensor ** out_mean = nullptr, // [T] (optional)
        ggml_tensor ** out_var  = nullptr  // [T] (optional)
) {
    GGML_ASSERT(x_ftc1);
    GGML_ASSERT(w_c);
    GGML_ASSERT(x_ftc1->ne[3] == 1);
    GGML_ASSERT(w_c->ne[0] == x_ftc1->ne[2]);

    const int64_t F = x_ftc1->ne[0];
    const int64_t T = x_ftc1->ne[1];
    const int64_t C = x_ftc1->ne[2];
    const float group_size = (float) (F*C);

    // HF computes stats in fp32 for numerical stability.
    ggml_tensor * x = x_ftc1;
    if (x->type != GGML_TYPE_F32) {
        x = ggml_cast(ctx0, x, GGML_TYPE_F32);
    }

    ggml_tensor * w = w_c;
    if (w->type != GGML_TYPE_F32) {
        w = ggml_cast(ctx0, w, GGML_TYPE_F32);
    }

    // Flatten (F, C) -> (F*C) while keeping T as the 2nd dim:
    //   x: [F, T, C, 1] -> [F, C, T, 1] -> [F*C, T]
    //
    // NOTE: ggml_permute() params specify where each *input* axis lands in the output:
    //   axis0 = new index for old axis 0, axis1 = new index for old axis 1, ...
    // To get [F, C, T, 1] (old0, old2, old1, old3) from [F, T, C, 1] we need:
    //   old0(F)->new0, old1(T)->new2, old2(C)->new1, old3->new3 => (0, 2, 1, 3).
    ggml_tensor * x_fct = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)); // [F, C, T, 1]
    ggml_tensor * a = ggml_reshape_2d(ctx0, x_fct, F*C, T);                   // [F*C, T]

    // sum_values_at_t: sum over (F, C) for each time step.
    // ggml_sum_rows returns shape [1, T]; view it as a length-T vector so ggml_cumsum runs over time (ne0).
    ggml_tensor * sum_at_t = ggml_sum_rows(ctx0, a);                  // [1, T]
    ggml_tensor * sum_vec  = ggml_reshape_1d(ctx0, sum_at_t, T);      // [T]
    ggml_tensor * cum_sum  = ggml_cumsum(ctx0, sum_vec);              // [T]

    // cum_count_elements: (t+1) * (F*C)
    ggml_tensor * count = ggml_arange(ctx0, 1.0f, (float) (T + 1), 1.0f); // [T]
    count = ggml_scale(ctx0, count, group_size);                          // [T]

    // cum_mean: [T]
    ggml_tensor * mean = ggml_div(ctx0, cum_sum, count);

    // squared diffs at each time step (uses the *cumulative mean at that time step*, matching HF):
    //   diff: [F*C, T]
    ggml_tensor * mean_rep = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, mean, 1, T), a);
    ggml_tensor * diff     = ggml_sub(ctx0, a, mean_rep);

    // sum_sq_diff_at_t: [1, T] -> [T], then cumulative sum over time.
    ggml_tensor * sq_sum_at_t = ggml_sum_rows(ctx0, ggml_mul(ctx0, diff, diff)); // [1, T]
    ggml_tensor * sq_sum_vec  = ggml_reshape_1d(ctx0, sq_sum_at_t, T);           // [T]
    ggml_tensor * cum_sq_sum  = ggml_cumsum(ctx0, sq_sum_vec);                   // [T]

    // cum_variance: [T]
    ggml_tensor * var = ggml_div(ctx0, cum_sq_sum, count);

    if (out_mean) {
        *out_mean = mean;
    }
    if (out_var) {
        *out_var = var;
    }

    // Normalize: (x - mean) / sqrt(var + eps)
    // Construct a scalar eps without requiring tensor data allocation (ctx0 is no_alloc).
    ggml_tensor * eps_scalar = ggml_arange(ctx0, eps, eps + 1.0f, 1.0f); // [1] (scalar)
    ggml_tensor * var_eps = ggml_add1(ctx0, var, eps_scalar);            // [T]
    ggml_tensor * denom   = ggml_sqrt(ctx0, ggml_repeat(ctx0, ggml_reshape_2d(ctx0, var_eps, 1, T), a)); // [F*C, T]
    ggml_tensor * norm_a  = ggml_div(ctx0, diff, denom); // [F*C, T]

    // Reshape back to [T, F, C, 1] and apply per-channel scale.
    ggml_tensor * norm_fct = ggml_reshape_4d(ctx0, norm_a, F, C, T, 1); // [F, C, T, 1]
    ggml_tensor * norm_cft = ggml_cont(ctx0, ggml_permute(ctx0, norm_fct, 1, 0, 2, 3)); // [C, F, T, 1]
    ggml_tensor * scaled   = ggml_mul(ctx0, norm_cft, w); // [C, F, T, 1]

    // Back to original layout [F, T, C, 1]
    return ggml_cont(ctx0, ggml_permute(ctx0, scaled, 2, 0, 1, 3)); // [F, T, C, 1]
}

static ggml_tensor * gemma3na_build_audio_hard_embedding(
        ggml_context * ctx0,
        const clip_model & model,
        const float eps,
        const int64_t tok_idx
) {
    // embed_audio (hard tokens) = post_norm( projection( hard_norm( embedding(token_id) ) ) )
    GGML_ASSERT(model.mm_input_proj_w); // mm.a.embedding.weight
    GGML_ASSERT(model.mm_input_norm_w); // mm.a.hard_embedding_norm.weight
    GGML_ASSERT(model.mm_0_w);          // mm.a.embedding_projection.weight

    ggml_tensor * emb_w = model.mm_input_proj_w;
    GGML_ASSERT(emb_w->ne[0] == G3NA_N_CHANNELS);
    GGML_ASSERT(emb_w->ne[1] > 0);
    GGML_ASSERT(tok_idx >= 0);
    GGML_ASSERT(tok_idx < emb_w->ne[1]);

    ggml_tensor * tok_emb = ggml_view_1d(ctx0, emb_w, emb_w->ne[0], tok_idx * emb_w->nb[1]); // [1536]
    tok_emb = ggml_reshape_2d(ctx0, tok_emb, tok_emb->ne[0], 1); // [1536, 1]

    // hard_embedding_norm
    tok_emb = ggml_rms_norm(ctx0, tok_emb, eps);
    tok_emb = ggml_mul(ctx0, tok_emb, model.mm_input_norm_w); // [1536]

    // embedding_projection
    tok_emb = ggml_mul_mat(ctx0, model.mm_0_w, tok_emb); // [2048, 1]

    // embedding_post_projection_norm (with_scale=false => RMS norm only)
    tok_emb = ggml_rms_norm(ctx0, tok_emb, eps);

    return tok_emb; // [2048, 1]
}

ggml_cgraph * clip_graph_gemma3na::build() {
    ggml_tensor * inp = build_inp_raw(1);
    cb(inp, "input", -1);

    // Create position embedding input for relative position attention
    // Shape: [G3NA_MAX_SPAN, G3NA_N_CHANNELS] = [13, 1536]
    // This will be filled with sinusoidal embeddings at runtime
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, G3NA_N_CHANNELS, G3NA_MAX_SPAN);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);
    ggml_build_forward_expand(gf, pos_emb);

    // Input `inp_raw` is shaped as [F=128, T, 1] (freq-major).
    // Reshape for conv2d: [width, height, channels, batch] = [F, T, 1, 1]
    auto * cur = ggml_reshape_4d(ctx0, inp, inp->ne[0], inp->ne[1], inp->ne[2], 1);
    cb(cur, "input_4d", -1);

    // Subsample conv projection
    // PyTorch uses manual padding before each conv:
    //   manual_padding = (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom) = (1, 1, 0, 2)
    // F.pad(x, (1, 1, 0, 2)) pads: last dim (F=mel) by (1,1), second-to-last (T=time) by (0,2)
    // ggml_pad_ext: lp0, rp0 for dim0; lp1, rp1 for dim1, etc.
    // Our tensor is [F, T, C, B], so:
    //   dim0 = F: pad (1, 1) - left/right on freq
    //   dim1 = T: pad (0, 2) - top/bottom on time

    // Conv0: weight [3, 3, 1, 128] - 1 input channel, 128 output channels
    // PyTorch: Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), bias=False) with manual_padding
    // PyTorch manual_padding = (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom) = (1, 1, 0, 2)
    // F.pad applies: last dim (F=mel) gets (1,1), second-to-last (T=time) gets (0,2)
    // ggml tensor is [time, mel, channel, batch], so:
    //   dim0 (time) needs (0, 2) - F.pad's (top, bottom) on T
    //   dim1 (mel) needs (1, 1)  - F.pad's (left, right) on F
    if (model.pre_encode_conv_X_w[0]) {
        cur = ggml_pad_ext(ctx0, cur, 1, 1, 0, 2, 0, 0, 0, 0);
        cb(cur, "conv0_padded", -1);

        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], cur, 2, 2, 0, 0, 1, 1);
        cb(cur, "conv0_out", -1);
        // Output: [(128+2-3)/2+1, (T+2-3)/2+1, 128, 1] = [64, (T+1)/2, 128, 1]

        if (model.pre_encode_conv_X_b[0]) {
            // True Gemma3nAudioCumulativeGroupNorm (cumulative over time, reduce over freq+channels).
            ggml_tensor * conv0_cum_mean = nullptr;
            ggml_tensor * conv0_cum_var  = nullptr;
            cur = gemma3na_cumulative_group_norm(ctx0, cur, model.pre_encode_conv_X_b[0], 1e-3f, &conv0_cum_mean, &conv0_cum_var);
            cb(conv0_cum_mean, "conv0_cum_mean", -1);
            cb(conv0_cum_var,  "conv0_cum_var",  -1);
            cb(cur, "conv0_norm", -1);
        }
        cur = ggml_relu(ctx0, cur);
        cb(cur, "conv0_norm_relu", -1);
    }

    // Conv1: weight [3, 3, 128, 32] - 128 input channels, 32 output channels
    // PyTorch: Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2), bias=False) with manual_padding
    // Same padding logic as conv0: dim0 (freq) = (1,1), dim1 (time) = (0,2)
    if (model.pre_encode_conv_X_w[1]) {
        cur = ggml_pad_ext(ctx0, cur, 1, 1, 0, 2, 0, 0, 0, 0);
        cb(cur, "conv1_padded", -1);

        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[1], cur, 2, 2, 0, 0, 1, 1);
        cb(cur, "conv1_out", -1);
        // Output: [(64+2-3)/2+1, (T'+2-3)/2+1, 32, 1] = [32, (T'+1)/2, 32, 1]

        if (model.pre_encode_conv_X_b[1]) {
            cur = gemma3na_cumulative_group_norm(ctx0, cur, model.pre_encode_conv_X_b[1], 1e-3f);
            cb(cur, "conv1_norm", -1);
        }
        cur = ggml_relu(ctx0, cur);
        cb(cur, "conv1_norm_relu", -1);
    }

    // Flatten: After conv2d output is [F=32, T'', C=32, 1]
    // where T'' ≈ (T+1)/4 after two stride-2 convs
    // We want [features, time] = [32*32, T''] = [1024, T'']
    //
    // PyTorch: after conv_1 shape is [B, C, T, F] = [1, 32, 325, 32]
    // PyTorch flatten: permute(0,2,3,1) -> [B, T, F, C] -> reshape [B, T, F*C]
    // In row-major flatten of [F, C]: element [f, c] is at index i = f * C + c
    // So: f = i // C = i // 32, c = i % C = i % 32  (c varies fastest)
    //
    // ggml after conv2d: [F, T, C, B] = [32, 325, 32, 1]
    // To match PyTorch order (c varies fastest), we need:
    //   permute(2, 0, 1, 3): [F, T, C, B] -> [C, F, T, B]
    // After reshape [C*F, T], in column-major: element [c, f] at index i = c + f * C
    // So: c = i % C = i % 32, f = i // C = i // 32  (c varies fastest - matches PyTorch!)
    {
        int64_t f_dim = cur->ne[0];  // freq after conv = 32
        int64_t t_dim = cur->ne[1];  // time after conv = 325
        int64_t c_dim = cur->ne[2];  // channels = 32

        // Permute [F, T, C, 1] -> [C, F, T, 1] to match PyTorch flatten order
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));  // [C, F, T, 1]
        cur = ggml_reshape_2d(ctx0, cur, c_dim * f_dim, t_dim);  // [1024, T'']
        cb(cur, "flattened", -1);
    }

    // Input projection: weight [1024, 1536]
    // ggml_mul_mat computes w^T @ x, so [1024, 1536]^T @ [1024, T/4] = [1536, T/4]
    if (model.pre_encode_out_w) {
        cur = ggml_mul_mat(ctx0, model.pre_encode_out_w, cur);
        cb(cur, "input_proj", -1);
    }
    // cur shape: [1536, T/4]

    // Conformer layers
    // All operations maintain [hidden, time] = [1536, T/4] format
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];
        auto * residual = cur;

        cb(cur, "layer_input", il);

        // FFW layer start (first half-step FFN)
        // cur: [1536, T/4]
        {
            ggml_tensor * ffn_inp = cur;

            // Pre-norm: RMS norm along feature dimension (dim 0)
            if (layer.ff_norm_w) {
                ffn_inp = ggml_rms_norm(ctx0, ffn_inp, eps);
                ffn_inp = ggml_mul(ctx0, ffn_inp, layer.ff_norm_w);  // [1536] broadcast
                cb(ffn_inp, "ffw_start_pre_norm", il);
            }

            // FFN up: weight [1536, 6144], input [1536, T/4] -> [6144, T/4]
            if (layer.ff_up_w) {
                ffn_inp = ggml_mul_mat(ctx0, layer.ff_up_w, ffn_inp);
                ffn_inp = ggml_silu(ctx0, ffn_inp);
                cb(ffn_inp, "ffw_start_up", il);
            }

            // FFN down: weight [6144, 1536], input [6144, T/4] -> [1536, T/4]
            if (layer.ff_down_w) {
                ffn_inp = ggml_mul_mat(ctx0, layer.ff_down_w, ffn_inp);
                cb(ffn_inp, "ffw_start_down", il);
            }

            // Post-norm
            if (layer.ln_1_w) {
                ffn_inp = ggml_rms_norm(ctx0, ffn_inp, eps);
                ffn_inp = ggml_mul(ctx0, ffn_inp, layer.ln_1_w);
                cb(ffn_inp, "ffw_start_post_norm", il);
            }

            // Residual with factor 0.5
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, ffn_inp, 0.5f));
        }

        // Self-attention with relative position encoding
        // cur: [1536, T/4]
        {
            cur = residual;

            // Pre-norm
            if (layer.norm_conv_w) {
                cur = ggml_rms_norm(ctx0, cur, eps);
                cur = ggml_mul(ctx0, cur, layer.norm_conv_w);
                cb(cur, "attn_pre_norm", il);
            }

            // Q, K, V projections: weight [1536, 1536], input [1536, T/4] -> [1536, T/4]
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);

            // Per-dim scale for Q
            // PyTorch: per_dim_scale_sp = softplus(self.per_dim_scale)
            //          query_states = query_states * self.q_scale * per_dim_scale_sp
            // where q_scale = head_dim^(-0.5) / softplus(0) = 1/sqrt(192) / ln(2) ≈ 0.104
            if (layer.q_norm) {
                // q_norm stores the raw per_dim_scale values, need to apply softplus
                ggml_tensor * per_dim_scale_sp = ggml_softplus(ctx0, layer.q_norm);

                // q_scale = 1/sqrt(head_dim) / ln(2) = 1/sqrt(192) / 0.6931 ≈ 0.1042
                const float q_scale = 1.0f / (sqrtf((float)d_head) * logf(2.0f));
                per_dim_scale_sp = ggml_scale(ctx0, per_dim_scale_sp, q_scale);

                Qcur = ggml_mul(ctx0, Qcur, per_dim_scale_sp);
                cb(Qcur, "q_per_dim_scale", il);
            }

            // cur shape: [1536, T/4] = [n_head * d_head, n_tokens]
            const int64_t n_tokens = cur->ne[1];

            // Reshape for multi-head attention
            // [1536, T/4] -> [d_head, n_head, T/4] = [192, 8, T/4]
            ggml_tensor * Q = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_tokens);
            ggml_tensor * K = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_tokens);
            ggml_tensor * V = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_tokens);

            // Permute for batched attention
            // Q, K: [d_head, n_head, n_tokens] -> [d_head, n_tokens, n_head]
            Qcur = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Kcur = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            // V: [d_head, n_head, n_tokens] -> [n_tokens, d_head, n_head]
            Vcur = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));

            // term_ac: Q @ K^T (content-based attention)
            // Kcur: [d_head, n_tokens, n_head]
            // Qcur: [d_head, n_tokens, n_head]
            // KQ = Kcur^T @ Qcur = [n_tokens, n_tokens, n_head]
            ggml_tensor * term_ac = ggml_mul_mat(ctx0, Kcur, Qcur);
            cb(term_ac, "term_ac", il);

            // Relative position encoding (term_bd)
            // pos_emb: [1536, 13] (sinusoidal embeddings for positions [12, 11, ..., 0])
            // Project through pos_proj (layer.k_norm): [1536, 1536]^T @ [1536, 13] = [1536, 13]
            // Then reshape to [d_head, n_head, 13] and compute Q @ sin_emb^T
            ggml_tensor * term_bd = nullptr;
            if (layer.k_norm) {
                // Project position embeddings through pos_proj
                // pos_emb: [1536, 13] -> projected: [1536, 13]
                ggml_tensor * proj_pos = ggml_mul_mat(ctx0, layer.k_norm, pos_emb);
                cb(proj_pos, "proj_pos_emb", il);

                // Reshape for multi-head: [1536, 13] -> [d_head, n_head, 13] = [192, 8, 13]
                proj_pos = ggml_reshape_3d(ctx0, proj_pos, d_head, n_head, G3NA_MAX_SPAN);

                // Permute: [d_head, n_head, 13] -> [d_head, 13, n_head]
                proj_pos = ggml_cont(ctx0, ggml_permute(ctx0, proj_pos, 0, 2, 1, 3));
                cb(proj_pos, "proj_pos_reshaped", il);

                // Compute term_bd = Q @ sin_emb^T
                // Qcur: [d_head, n_tokens, n_head]
                // proj_pos: [d_head, 13, n_head]
                // Result: [13, n_tokens, n_head]
                term_bd = ggml_mul_mat(ctx0, proj_pos, Qcur);
                cb(term_bd, "term_bd_raw", il);

                // Apply relative shift to convert term_bd to attention bias matrix
                // term_bd shape: [13, n_tokens, n_head] = [pos_len, q_len, h]
                // pos_idx 0 -> relative position 12
                // pos_idx 12 -> relative position 0
                //
                // For attention[q, k], we need term_bd[12-(q-k)] when 0 <= q-k <= 12
                // Since Gemma3n has max_forward=0, positions k > q get no bias
                //
                // Apply relative shift following conformer.cpp pattern
                // The goal is to create a Toeplitz matrix where:
                // result[k, q, h] = term_bd[12 - (q - k), q, h] for valid positions
                {
                    // term_bd: [13, n_tokens, n_head] = [pos_len, q_len, h]
                    const int64_t pos_len = term_bd->ne[0];  // 13
                    const int64_t q_len = term_bd->ne[1];    // n_tokens
                    const int64_t h = term_bd->ne[2];        // n_head

                    // Step 1: Pad on dim 0 (pos dimension) by 1 on the right
                    term_bd = ggml_pad(ctx0, term_bd, 1, 0, 0, 0);
                    cb(term_bd, "term_bd_padded", il);
                    // term_bd: [14, n_tokens, n_head]

                    // Step 2: Roll by 1 on dim 0 to shift positions
                    term_bd = ggml_roll(ctx0, term_bd, 1, 0, 0, 0);
                    cb(term_bd, "term_bd_rolled", il);

                    // Step 3: Reshape to [q_len, pos_len+1, h] to prepare for diagonal extraction
                    term_bd = ggml_reshape_3d(ctx0, term_bd, q_len, pos_len + 1, h);
                    cb(term_bd, "term_bd_reshaped", il);

                    // Step 4: View to extract the diagonal structure
                    // Skip the first row and take min(pos_len, q_len) rows
                    int64_t take_len = (pos_len < q_len) ? pos_len : q_len;
                    term_bd = ggml_view_3d(ctx0, term_bd,
                                           q_len, take_len, h,
                                           term_bd->nb[1], term_bd->nb[2],
                                           term_bd->nb[0] * q_len);  // skip first q_len elements (one row)
                    term_bd = ggml_cont_3d(ctx0, term_bd, take_len, q_len, h);
                    cb(term_bd, "term_bd_shifted", il);
                    // term_bd: [take_len, q_len, h] = [min(13, n_tokens), n_tokens, h]

                    // Step 5: If take_len < n_tokens, we need to pad for the remaining positions
                    // These are the forward-looking positions (k > q) which get zero bias
                    if (take_len < q_len) {
                        // Pad on dim 0 to extend to n_tokens
                        int64_t pad_amount = q_len - take_len;
                        term_bd = ggml_pad(ctx0, term_bd, pad_amount, 0, 0, 0);
                        // Now we need to roll the zeros to the front (for k > q)
                        // Actually, looking at the attention matrix [n_tokens, n_tokens],
                        // the upper triangle (k > q) should get zero bias
                        // After reshape, dim 0 corresponds to key positions
                        // We want zeros at the end (large key positions for each query)
                        // which is what pad gives us, so no roll needed
                        cb(term_bd, "term_bd_pad_forward", il);
                    }
                    // term_bd: [n_tokens, n_tokens, n_head]
                }

                cb(term_bd, "term_bd_final", il);
            }

            // Combine term_ac and term_bd
            ggml_tensor * KQ = term_ac;
            if (term_bd) {
                KQ = ggml_add(ctx0, term_ac, term_bd);
            }
            cb(KQ, "attn_logits", il);

            // Apply attention logit softcap (50.0)
            const float softcap = 50.0f;
            KQ = ggml_scale(ctx0, KQ, 1.0f / softcap);
            KQ = ggml_tanh(ctx0, KQ);
            KQ = ggml_scale(ctx0, KQ, softcap);
            cb(KQ, "attn_logits_softcap", il);

            // Apply Gemma3n local causal attention mask.
            //
            // HF reference (Gemma3nAudioAttention):
            //   - conf_attention_chunk_size = 12
            //   - conf_attention_context_left = 13  -> max_past_horizon = 12
            //   - conf_attention_context_right = 0
            //
            // This results in a fixed local causal window of length 13:
            //   allow keys k in [q - 12, q] for each query q.
            //
            // In ggml KQ layout, dim0 = key index (k), dim1 = query index (q).
            {
                const int max_past_horizon = G3NA_MAX_SPAN - 1; // 12

                // Mask future keys (k > q)
                KQ = ggml_diag_mask_inf(ctx0, KQ, /*n_past*/ 0);

                // Mask keys too far in the past (k < q - max_past_horizon) by transposing,
                // applying the same diag mask, then transposing back.
                ggml_tensor * KQ_t = ggml_cont(ctx0, ggml_permute(ctx0, KQ, 1, 0, 2, 3));
                KQ_t = ggml_diag_mask_inf(ctx0, KQ_t, /*n_past*/ max_past_horizon);
                KQ = ggml_cont(ctx0, ggml_permute(ctx0, KQ_t, 1, 0, 2, 3));

                cb(KQ, "attn_logits_masked", il);
            }

            // Softmax
            KQ = ggml_soft_max(ctx0, KQ);
            cb(KQ, "attn_weights", il);

            // Attention output: attn @ V
            // Vcur: [n_tokens, d_head, n_head]
            // KQ:   [n_tokens, n_tokens, n_head] (keys, queries, head)
            // KQV = Vcur^T @ KQ -> [d_head, n_tokens, n_head]
            ggml_tensor * KQV = ggml_mul_mat(ctx0, Vcur, KQ);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);                     // [d_head, n_head, n_tokens]
            KQV = ggml_cont_2d(ctx0, KQV, d_head * n_head, n_tokens);        // [1536, T/4]

            // Output projection: weight [1536, 1536]
            cur = ggml_mul_mat(ctx0, layer.o_w, KQV);
            cb(cur, "attn_output", il);

            // Post-norm
            if (layer.norm_conv_b) {
                cur = ggml_rms_norm(ctx0, cur, eps);
                cur = ggml_mul(ctx0, cur, layer.norm_conv_b);
                cb(cur, "attn_post_norm", il);
            }

            residual = ggml_add(ctx0, residual, cur);
        }

        // LConv1D (Lightweight Convolution)
        // cur: [1536, T/4]
        {
            cur = residual;

            // Pre-norm
            if (layer.linear_pos_w) {
                cur = ggml_rms_norm(ctx0, cur, eps);
                cur = ggml_mul(ctx0, cur, layer.linear_pos_w);
                cb(cur, "lconv_pre_norm", il);
            }

            // Linear start with GLU: weight [1536, 3072]
            // [1536, 3072]^T @ [1536, T/4] = [3072, T/4]
            if (layer.conv_pw1_w) {
                cur = ggml_mul_mat(ctx0, layer.conv_pw1_w, cur);
                // GLU: split [3072, T/4] into [1536, T/4] and [1536, T/4]
                int64_t d = cur->ne[0] / 2;  // 1536
                int64_t n_tok = cur->ne[1];
                // First half: values, second half: gates
                ggml_tensor * val = ggml_view_2d(ctx0, cur, d, n_tok, cur->nb[1], 0);
                ggml_tensor * gate = ggml_view_2d(ctx0, cur, d, n_tok, cur->nb[1], d * ggml_element_size(cur));
                gate = ggml_sigmoid(ctx0, gate);
                cur = ggml_mul(ctx0, val, gate);
                // Make contiguous and transpose for conv: [1536, T/4] -> [T/4, 1536]
                cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
                cb(cur, "lconv_glu", il);
            }
            // cur: [T/4, 1536] after transpose

            // Depthwise conv1d: weight [5, 1, 1536]
            // ssm_conv expects:
            //   input sx: 3D [n_t + d_conv - 1, d_inner, n_s]
            //   kernel c: 2D [d_conv, d_inner]
            //   output: 3D [d_inner, n_t, n_s]
            if (layer.conv_dw_w) {
                // Reshape kernel from [5, 1, 1536] to [5, 1536]
                ggml_tensor * conv_w = ggml_reshape_2d(ctx0, layer.conv_dw_w,
                                                       layer.conv_dw_w->ne[0],  // 5
                                                       layer.conv_dw_w->ne[2]); // 1536
                // Metal ssm_conv requires F32 kernel
                conv_w = ggml_cast(ctx0, conv_w, GGML_TYPE_F32);

                // Save original time dimension for later
                int64_t orig_time = cur->ne[0];

                // Add batch dimension: [T/4, 1536] -> [T/4, 1536, 1]
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], cur->ne[1], 1);

                // Pad for causal conv with kernel 5
                cur = ggml_pad(ctx0, cur, 4, 0, 0, 0);
                cur = ggml_roll(ctx0, cur, 4, 0, 0, 0);
                cur = ggml_pad(ctx0, cur, 4, 0, 0, 0);

                cur = ggml_ssm_conv(ctx0, cur, conv_w);
                // Output: [d_inner, n_t, n_s] = [1536, T/4 + 4, 1]
                cb(cur, "lconv_dw", il);

                // Remove batch dimension: [1536, T+4, 1] -> [1536, T+4]
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1]);

                // Truncate to original time dimension: take first orig_time frames
                cur = ggml_view_2d(ctx0, cur, cur->ne[0], orig_time, cur->nb[1], 0);
                cur = ggml_cont(ctx0, cur);
            }
            // cur: [1536, T/4]

            // Conv norm + SiLU
            if (layer.conv_norm_w) {
                cur = ggml_rms_norm(ctx0, cur, eps);
                cur = ggml_mul(ctx0, cur, layer.conv_norm_w);
            }
            cur = ggml_silu(ctx0, cur);
            // cur: [1536, T/4]

            // Linear end: weight [1536, 1536]
            if (layer.conv_pw2_w) {
                cur = ggml_mul_mat(ctx0, layer.conv_pw2_w, cur);
                cb(cur, "lconv_end", il);
            }

            residual = ggml_add(ctx0, residual, cur);
        }

        // FFW layer end (second half-step FFN)
        // cur: [1536, T/4]
        {
            cur = residual;

            // Pre-norm
            if (layer.ff_norm_1_w) {
                cur = ggml_rms_norm(ctx0, cur, eps);
                cur = ggml_mul(ctx0, cur, layer.ff_norm_1_w);
                cb(cur, "ffw_end_pre_norm", il);
            }

            // FFN up: weight [1536, 6144]
            if (layer.ff_up_1_w) {
                cur = ggml_mul_mat(ctx0, layer.ff_up_1_w, cur);
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffw_end_up", il);
            }

            // FFN down: weight [6144, 1536]
            if (layer.ff_down_1_w) {
                cur = ggml_mul_mat(ctx0, layer.ff_down_1_w, cur);
                cb(cur, "ffw_end_down", il);
            }

            // Post-norm
            if (layer.ln_2_w) {
                cur = ggml_rms_norm(ctx0, cur, eps);
                cur = ggml_mul(ctx0, cur, layer.ln_2_w);
                cb(cur, "ffw_end_post_norm", il);
            }

            // Residual with factor 0.5
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, 0.5f));
        }

        // Block norm
        if (layer.ls_1_w) {
            residual = ggml_rms_norm(ctx0, residual, eps);
            residual = ggml_mul(ctx0, residual, layer.ls_1_w);
            cb(residual, "block_norm", il);
        }

        cur = residual;
    }

    // Temporal reduction (HF: `audio_encodings = audio_encodings[:, ::4]`)
    // HF applies this reduction inside the audio encoder (before embed_audio).
    {
        const int conf_reduction_factor = 4;
        const int64_t n_tokens_in  = cur->ne[1];
        const int64_t n_tokens_out = (n_tokens_in + conf_reduction_factor - 1) / conf_reduction_factor;
        cur = ggml_view_2d(ctx0, cur, cur->ne[0], n_tokens_out, cur->nb[1] * conf_reduction_factor, 0);
        cur = ggml_cont(ctx0, cur);
        cb(cur, "temporal_reduction", -1);
    }

    // Audio embedding projection (Gemma3nMultimodalEmbedder.forward for soft tokens)
    // cur: [1536, T/16] after temporal reduction
    //
    // PyTorch flow for inputs_embeds (soft tokens = audio features):
    // 1. soft_embedding_norm(inputs_embeds) - RMS norm with [1536] weight
    // 2. embedding_projection(emb_norm) - Linear [1536 -> 2048]
    // 3. embedding_post_projection_norm(emb_norm_proj) - scalar norm (weight=1.0)
    //
    // Note: hard_embedding_norm is only used for token IDs, not soft tokens!

    // Step 1: Soft embedding norm (for soft tokens / audio features)
    if (model.mm_soft_emb_norm_w) {
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);  // [1536]
        cb(cur, "soft_embedding_norm", -1);
    }

    // Step 2: Embedding projection: [1536, 2048]^T @ [1536, T] = [2048, T]
    if (model.mm_0_w) {
        cur = ggml_mul_mat(ctx0, model.mm_0_w, cur);
        cb(cur, "embedding_projection", -1);
    }
    // cur: [2048, T/4]

    // Step 3: Post projection norm (embedding_post_projection_norm, weight=1.0 scalar)
    // Since the weight is just 1.0, this is effectively just an RMS norm
    cur = ggml_rms_norm(ctx0, cur, eps);
    cb(cur, "embedding_post_projection_norm", -1);

    // HF pads audio features to a fixed 188 soft tokens (30s chunk) using the last embed_audio vocab embedding.
    // <end_of_audio> itself is injected as a normal vocab token, not via the audio encoder output.
    {
        const int64_t n_tokens = cur->ne[1];
        if (n_tokens < G3NA_AUDIO_SOFT_TOKENS) {
            // audio_padding_embs = embed_audio(input_ids=vocab_size-1)
            ggml_tensor * pad_emb = gemma3na_build_audio_hard_embedding(ctx0, model, eps, model.mm_input_proj_w->ne[1] - 1); // [2048, 1]
            cb(pad_emb, "audio_padding_embedding", -1);

            const int64_t n_extra = G3NA_AUDIO_SOFT_TOKENS - n_tokens;
            ggml_tensor * shape = ggml_new_tensor_2d(ctx0, pad_emb->type, pad_emb->ne[0], n_extra);
            ggml_tensor * extra = ggml_repeat(ctx0, pad_emb, shape); // [2048, n_extra]

            cur = ggml_concat(ctx0, cur, extra, 1);
            cur = ggml_cont(ctx0, cur);
            cb(cur, "output_padded", -1);
        } else if (n_tokens > G3NA_AUDIO_SOFT_TOKENS) {
            cur = ggml_view_2d(ctx0, cur, cur->ne[0], G3NA_AUDIO_SOFT_TOKENS, cur->nb[1], 0);
            cur = ggml_cont(ctx0, cur);
            cb(cur, "output_truncated", -1);
        }
        GGML_ASSERT(cur->ne[1] == G3NA_AUDIO_TOKENS_OUT);
    }

    cb(cur, "output", -1);
    ggml_build_forward_expand(gf, cur);

    return gf;
}
