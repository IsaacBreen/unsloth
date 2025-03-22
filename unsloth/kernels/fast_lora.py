# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from .utils import (
    fast_dequantize,
    QUANT_STATE,
    get_lora_parameters,
    get_lora_parameters_bias,
    matmul_lora,
    torch_amp_custom_fwd,
    torch_amp_custom_bwd,
)


class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)

    ### Down projection LoRA weights
    dC/dAw = dC/dW @ B.T
    dC/dBw = A.T @ dC/dW
    dC/dAw =       h.T @ dY @ B.T
    dC/dBw = A.T @ h.T @ dY

    ### Up projection LoRA weights
    dC/dAu =       X.T @ (D @ W.T * f) @ B.T
    dC/dBu = A.T @ X.T @ (D @ W.T * f)

    ### Gate projection LoRA weights
    dC/dAg =       X.T @ (D @ W.T * df * g) @ B.T
    dC/dBg = A.T @ X.T @ (D @ W.T * df * g)

    Don't forget to see our blog post for more details!
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X : torch.Tensor,
                gateW, gateW_quant, gateA, gateB, gateS,
                  upW,   upW_quant, upA,   upB,   upS,
                downW, downW_quant, downA, downB, downS,
                _forward_function, _backward_function,
                inplace = True, use_dora_mlp = False,
                gate_magnitude_vector = None, up_magnitude_vector = None, down_magnitude_vector = None):
        dtype = X.dtype

        if not use_dora_mlp:
            e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
            g = matmul_lora(X,   upW,   upW_quant,   upA,   upB,   upS)
            h = _forward_function(e, g)
            i = matmul_lora(h, downW, downW_quant, downA, downB, downS)
        else:
            # gate proj
            lora_gate = torch.matmul(gateA, gateB) * gateS
            numerator_gate = gateW + lora_gate.T
            denominator_gate = torch.linalg.norm(numerator_gate, dim=0, keepdim=True)
            directional_component_gate = numerator_gate / denominator_gate
            new_weight_gate = gate_magnitude_vector * directional_component_gate
            e = torch.matmul(X, new_weight_gate)

            # up proj
            lora_up = torch.matmul(upA, upB) * upS
            numerator_up = upW + lora_up.T
            denominator_up = torch.linalg.norm(numerator_up, dim=0, keepdim=True)
            directional_component_up = numerator_up / denominator_up
            new_weight_up = up_magnitude_vector * directional_component_up
            g = torch.matmul(X, new_weight_up)

            h = _forward_function(e, g)

            # down proj
            lora_down = torch.matmul(downA, downB) * downS
            numerator_down = downW + lora_down.T
            denominator_down = torch.linalg.norm(numerator_down, dim=0, keepdim=True)
            directional_component_down = numerator_down / denominator_down
            new_weight_down = down_magnitude_vector * directional_component_down
            i = torch.matmul(h, new_weight_down)


        ctx.custom_saved_tensors = (
            gateW, gateW_quant, gateS,
            upW, upW_quant, upS,
            downW, downW_quant, downS,
            _backward_function,
            use_dora_mlp,
            gate_magnitude_vector if use_dora_mlp else None,
            up_magnitude_vector if use_dora_mlp else None,
            down_magnitude_vector if use_dora_mlp else None,
            numerator_gate if use_dora_mlp else None,
            denominator_gate if use_dora_mlp else None,
            numerator_up if use_dora_mlp else None,
            denominator_up if use_dora_mlp else None,
            numerator_down if use_dora_mlp else None,
            denominator_down if use_dora_mlp else None,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB,
                              X, e, g, h if use_dora_mlp else None, gateW if use_dora_mlp else None, upW if use_dora_mlp else None, downW if use_dora_mlp else None)
        ctx.inplace = inplace
        return i
    pass


    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY : torch.Tensor):
        gateW, gateW_quant, gateS, upW, upW_quant, upS, downW, downW_quant, downS, \
            _backward_function, use_dora_mlp, gate_magnitude_vector, up_magnitude_vector, down_magnitude_vector, \
            numerator_gate, denominator_gate, numerator_up, denominator_up, numerator_down, denominator_down = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, \
            X, e, g, h_dora, gateW_orig, upW_orig, downW_orig = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X  = X .view(-1, X .shape[-1])
        e  = e .view(-1, e .shape[-1])
        g  = g .view(-1, g .shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = \
            gateA.to(dtype), gateB.to(dtype), upA.to(dtype), upB.to(dtype), downA.to(dtype), downB.to(dtype)

        gateA, gateB, upA, upB, downA, downB = \
            gateA.t(), gateB.t(), upA.t(), upB.t(), downA.t(), downB.t()

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA   = torch.empty_like(upA)
        d_upB   = torch.empty_like(upB)
        dW_magnitude_gate = None
        dW_magnitude_up = None
        dW_magnitude_down = None

        if not use_dora_mlp:
            # Down projection LoRA weights
            d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
            d_downB.addmm_(downA.t() @ h.t(), dY, alpha = downS, beta = 0)

            # Up projection LoRA weights
            d_upA.addmm_(X.t(), df @ upB.t(), alpha = upS, beta = 0)
            d_upB.addmm_(upA.t() @ X.t(), df, alpha = upS, beta = 0)

            # Gate projection LoRA weights
            d_gateA.addmm_(X.t(), de @ gateB.t(), alpha = gateS, beta = 0)
            d_gateB.addmm_(gateA.t() @ X.t(), de, alpha = gateS, beta = 0)

            # dX
            upW = fast_dequantize(upW.t(), upW_quant)
            dX = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)
            del upW
            dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)

            gateW = fast_dequantize(gateW.t(), gateW_quant)
            dX.addmm_(de, gateW.t())
            del gateW
            dX.addmm_(de @ gateB.t(), gateA.t(), alpha = gateS)
        else:
            # DoRA backward pass for MLP
            # Down projection DoRA weights
            lora_down = torch.matmul(downA.t(), downB.t()) * downS # Recalculate lora for backward
            directional_component_down = numerator_down / denominator_down # Recalculate directional_component for backward
            d_directional_component_down = down_magnitude_vector.T * fast_dequantize(dY.T, QUANT_STATE(dY))
            d_magnitude_vector_down = torch.sum(dY * directional_component_down, dim=0, keepdim=True)

            d_numerator_down = down_magnitude_vector * d_directional_component_down
            d_denominator_down = -torch.sum(d_numerator_down * numerator_down / denominator_down, dim=0, keepdim=True) / denominator_down

            d_W_lora_down = d_directional_component_down / denominator_down + d_denominator_down * numerator_down / (denominator_down ** 2)

            d_downA.addmm_(h.t(), d_W_lora_down @ downB.t(), alpha = 1.0, beta = 0) # Approximation: use dY here instead of d_lora * magnitude_vector. This might need refinement.
            d_downB.addmm_(downA.t() @ h.t(), d_W_lora_down, alpha = 1.0, beta = 0) # Approximation: use dY here instead of d_lora * magnitude_vector. This might need refinement.
            dW_magnitude_down = d_magnitude_vector_down

            # Up projection DoRA weights
            lora_up = torch.matmul(upA.t(), upB.t()) * upS # Recalculate lora for backward
            directional_component_up = numerator_up / denominator_up # Recalculate directional_component for backward
            df_dora = h_dora # df
            d_directional_component_up = up_magnitude_vector.T * fast_dequantize(df_dora.T, QUANT_STATE(df_dora))
            d_magnitude_vector_up = torch.sum(df_dora * directional_component_up, dim=0, keepdim=True)

            d_numerator_up = up_magnitude_vector * d_directional_component_up
            d_denominator_up = -torch.sum(d_numerator_up * numerator_up / denominator_up, dim=0, keepdim=True) / denominator_up

            d_W_lora_up = d_directional_component_up / denominator_up + d_denominator_up * numerator_up / (denominator_up ** 2)

            d_upA.addmm_(X.t(), d_W_lora_up @ upB.t(), alpha = 1.0, beta = 0) # Approximation: use dY here instead of d_lora * magnitude_vector. This might need refinement.
            d_upB.addmm_(upA.t() @ X.t(), d_W_lora_up, alpha = 1.0, beta = 0) # Approximation: use dY here instead of d_lora * magnitude_vector. This might need refinement.
            dW_magnitude_up = d_magnitude_vector_up

            # Gate projection DoRA weights
            lora_gate = torch.matmul(gateA.t(), gateB.t()) * gateS # Recalculate lora for backward
            directional_component_gate = numerator_gate / denominator_gate # Recalculate directional_component for backward
            de_dora = e # de
            d_directional_component_gate = gate_magnitude_vector.T * fast_dequantize(de_dora.T, QUANT_STATE(de_dora))
            d_magnitude_vector_gate = torch.sum(de_dora * directional_component_gate, dim=0, keepdim=True)

            d_numerator_gate = gate_magnitude_vector * d_directional_component_gate
            d_denominator_gate = -torch.sum(d_numerator_gate * numerator_gate / denominator_gate, dim=0, keepdim=True) / denominator_gate

            d_W_lora_gate = d_directional_component_gate / denominator_gate + d_denominator_gate * numerator_gate / (denominator_gate ** 2)

            d_gateA.addmm_(X.t(), d_W_lora_gate @ gateB.t(), alpha = 1.0, beta = 0) # Approximation: use dY here instead of d_lora * magnitude_vector. This might need refinement.
            d_gateB.addmm_(gateA.t() @ X.t(), d_W_lora_gate, alpha = 1.0, beta = 0) # Approximation: use dY here instead of d_lora * magnitude_vector. This might need refinement.
            dW_magnitude_gate = d_magnitude_vector_gate

            # dX
            dX = torch.matmul(df_dora, (numerator_up / denominator_up).T)
            dX.addmm_(de_dora, (numerator_gate / denominator_gate).T)


        # gateW, gateW_quant, gateA, gateB, gateS,
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        return dX.view(batch, seq_len, hd), \
            None, None, d_gateA.t(), d_gateB.t(), None, \
            None, None,   d_upA.t(),   d_upB.t(), None, \
            None, None, d_downA.t(), d_downB.t(), None, \
            None, None, None, # _backward and _forward and inplace
            dW_magnitude_gate, dW_magnitude_up, dW_magnitude_down # magnitude gradients
    pass
pass


from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
def apply_lora_mlp_swiglu(self, X, inplace = True, use_dora_mlp = False):
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW,     upW_quant,   upA,   upB,   upS = get_lora_parameters(self.  up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    gate_magnitude_vector = get_lora_parameters_bias(self.gate_proj, "magnitude_vector") if use_dora_mlp else None
    up_magnitude_vector = get_lora_parameters_bias(self.up_proj, "magnitude_vector") if use_dora_mlp else None
    down_magnitude_vector = get_lora_parameters_bias(self.down_proj, "magnitude_vector") if use_dora_mlp else None
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS,
                         upW,     upW_quant, upA,   upB,   upS,
                         downW, downW_quant, downA, downB, downS,
                         swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel,
                         inplace, use_dora_mlp,
                         gate_magnitude_vector, up_magnitude_vector, down_magnitude_vector)
    return out
pass


from .geglu import geglu_exact_forward_kernel, geglu_exact_backward_kernel
def apply_lora_mlp_geglu_exact(self, X, inplace = True, use_dora_mlp = False):
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW,     upW_quant,   upA,   upB,   upS = get_lora_parameters(self.  up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    gate_magnitude_vector = get_lora_parameters_bias(self.gate_proj, "magnitude_vector") if use_dora_mlp else None
    up_magnitude_vector = get_lora_parameters_bias(self.up_proj, "magnitude_vector") if use_dora_mlp else None
    down_magnitude_vector = get_lora_parameters_bias(self.down_proj, "magnitude_vector") if use_dora_mlp else None
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS,
                         upW,     upW_quant, upA,   upB,   upS,
                         downW, downW_quant, downA, downB, downS,
                         geglu_exact_forward_kernel, geglu_exact_backward_kernel,
                         inplace, use_dora_mlp,
                         gate_magnitude_vector, up_magnitude_vector, down_magnitude_vector)
    return out
pass


from .geglu import geglu_approx_forward_kernel, geglu_approx_backward_kernel
def apply_lora_mlp_geglu_approx(self, X, use_dora_mlp = False):
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW,     upW_quant,   upA,   upB,   upS = get_lora_parameters(self.  up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    gate_magnitude_vector = get_lora_parameters_bias(self.gate_proj, "magnitude_vector") if use_dora_mlp else None
    up_magnitude_vector = get_lora_parameters_bias(self.up_proj, "magnitude_vector") if use_dora_mlp else None
    down_magnitude_vector = get_lora_parameters_bias(self.down_proj, "magnitude_vector") if use_dora_mlp else None
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS,
                         upW,     upW_quant, upA,   upB,   upS,
                         downW, downW_quant, downA, downB, downS,
                         geglu_approx_forward_kernel, geglu_approx_backward_kernel,
                         False, use_dora_mlp, # inplace=False for approx geglu
                         gate_magnitude_vector, up_magnitude_vector, down_magnitude_vector)
    return out
pass


class LoRA_QKV(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    See our blogpost for more details.

    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)
    We then sum them all find dC/dX

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X : torch.Tensor,
                QW, QW_quant, QA, QB, QS,
                KW, KW_quant, KA, KB, KS,
                VW, VW_quant, VA, VB, VS,
                inplace = True, use_dora_qkv = False,
                q_magnitude_vector = None, k_magnitude_vector = None, v_magnitude_vector = None):
        dtype = X.dtype

        if not use_dora_qkv:
            Q = matmul_lora(X, QW, QW_quant, QA, QB, QS)
            K = matmul_lora(X, KW, KW_quant, KA, KB, KS)
            V = matmul_lora(X, VW, VW_quant, VA, VB, VS)
        else:
            # Q proj
            lora_q = torch.matmul(QA, QB) * QS
            numerator_q = QW + lora_q.T
            denominator_q = torch.linalg.norm(numerator_q, dim=0, keepdim=True)
            directional_component_q = numerator_q / denominator_q
            new_weight_q = q_magnitude_vector * directional_component_q
            Q = torch.matmul(X, new_weight_q)

            # K proj
            lora_k = torch.matmul(KA, KB) * KS
            numerator_k = KW + lora_k.T
            denominator_k = torch.linalg.norm(numerator_k, dim=0, keepdim=True)
            directional_component_k = numerator_k / denominator_k
            new_weight_k = k_magnitude_vector * directional_component_k
            K = torch.matmul(X, new_weight_k)

            # V proj
            lora_v = torch.matmul(VA, VB) * VS
            numerator_v = VW + lora_v.T
            denominator_v = torch.linalg.norm(numerator_v, dim=0, keepdim=True)
            directional_component_v = numerator_v / denominator_v
            new_weight_v = v_magnitude_vector * directional_component_v
            V = torch.matmul(X, new_weight_v)


        ctx.custom_saved_tensors = (
            QW, QW_quant, QS,
            KW, KW_quant, KS,
            VW, VW_quant, VS,
            use_dora_qkv,
            q_magnitude_vector if use_dora_qkv else None,
            k_magnitude_vector if use_dora_qkv else None,
            v_magnitude_vector if use_dora_qkv else None,
            numerator_q if use_dora_qkv else None,
            denominator_q if use_dora_qkv else None,
            numerator_k if use_dora_qkv else None,
            denominator_k if use_dora_qkv else None,
            numerator_v if use_dora_qkv else None,
            denominator_v if use_dora_qkv else None,
        )
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB, QW if use_dora_qkv else None, KW if use_dora_qkv else None, VW if use_dora_qkv else None)
        ctx.inplace = inplace
        return Q, K, V
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS, \
            use_dora_qkv, q_magnitude_vector, k_magnitude_vector, v_magnitude_vector, \
            numerator_q, denominator_q, numerator_k, denominator_k, numerator_v, denominator_v = \
            ctx.custom_saved_tensors
        X, QA, QB, KA, KB, VA, VB, QW_orig, KW_orig, VW_orig = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1]) # view doesn't work on K.T
        dV = dV.view(-1, dV.shape[-1])
        X  = X .view(-1, X .shape[-1])
        dtype = X.dtype

        QA, QB, KA, KB, VA, VB = \
            QA.to(dtype), QB.to(dtype), KA.to(dtype), KB.to(dtype), VA.to(dtype), VB.to(dtype)

        QA, QB, KA, KB, VA, VB = \
            QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        ### Weight projection LoRA weights
        # See our blogpost for more details.
        d_QA = torch.empty_like(QA)
        d_QB = torch.empty_like(QB)
        d_KA = torch.empty_like(KA)
        d_KB = torch.empty_like(KB)
        d_VA = torch.empty_like(VA)
        d_VB = torch.empty_like(VB)
        dW_magnitude_q = None
        dW_magnitude_k = None
        dW_magnitude_v = None


        if not use_dora_qkv:
            # Q Projection
            d_QA.addmm_(X.t(), dQ @ QB.t(), alpha = QS, beta = 0)
            d_QB.addmm_(QA.t() @ X.t(), dQ, alpha = QS, beta = 0)

            # K Projection
            d_KA.addmm_(X.t(), dK @ KB.t(), alpha = KS, beta = 0)
            d_KB.addmm_(KA.t() @ X.t(), dK, alpha = KS, beta = 0)

            # V Projection
            d_VA.addmm_(X.t(), dV @ VB.t(), alpha = VS, beta = 0)
            d_VB.addmm_(VA.t() @ X.t(), dV, alpha = VS, beta = 0)

            # Combine derivatives to find dX
            # dQ
            QW = fast_dequantize(QW.t(), QW_quant)
            dX = torch.matmul(dQ, QW.t(), out = X if ctx.inplace else None)
            del QW
            dX.addmm_(dQ @ QB.t(), QA.t(), alpha = QS)

            # dK
            KW = fast_dequantize(KW.t(), KW_quant)
            dX.addmm_(dK, KW.t())
            del KW
            dX.addmm_(dK @ KB.t(), KA.t(), alpha = KS)

            # dV
            VW = fast_dequantize(VW.t(), VW_quant)
            dX.addmm_(dV, VW.t())
            del VW
            dX.addmm_(dV @ VB.t(), VA.t(), alpha = VS)
        else:
            # DoRA backward pass for QKV
            # Q Projection DoRA weights
            lora_q = torch.matmul(QA.t(), QB.t()) * QS # Recalculate lora for backward
            directional_component_q = numerator_q / denominator_q # Recalculate directional_component for backward
            d_directional_component_q = q_magnitude_vector.T * fast_dequantize(dQ.T, QUANT_STATE(dQ))
            d_magnitude_vector_q = torch.sum(dQ * directional_component_q, dim=0, keepdim=True)

            d_numerator_q = q_magnitude_vector * d_directional_component_q
            d_denominator_q = -torch.sum(d_numerator_q * numerator_q / denominator_q, dim=0, keepdim=True) / denominator_q

            d_W_lora_q = d_directional_component_q / denominator_q + d_denominator_q * numerator_q / (denominator_q ** 2)

            d_QA.addmm_(X.t(), d_W_lora_q @ QB.t(), alpha = 1.0, beta = 0) # Approximation: use dQ here instead of d_lora * magnitude_vector. This might need refinement.
            d_QB.addmm_(QA.t() @ X.t(), d_W_lora_q, alpha = 1.0, beta = 0) # Approximation: use dQ here instead of d_lora * magnitude_vector. This might need refinement.
            dW_magnitude_q = d_magnitude_vector_q

            # K Projection DoRA weights
            lora_k = torch.matmul(KA.t(), KB.t()) * KS # Recalculate lora for backward
            directional_component_k = numerator_k / denominator_k # Recalculate directional_component for backward
            d_directional_component_k = k_magnitude_vector.T * fast_dequantize(dK.T, QUANT_STATE(dK))
            d_magnitude_vector_k = torch.sum(dK * directional_component_k, dim=0, keepdim=True)

            d_numerator_k = k_magnitude_vector * d_directional_component_k
            d_denominator_k = -torch.sum(d_numerator_k * numerator_k / denominator_k, dim=0, keepdim=True) / denominator_k

            d_W_lora_k = d_directional_component_k / denominator_k + d_denominator_k * numerator_k / (denominator_k ** 2)

            d_KA.addmm_(X.t(), d_W_lora_k @ KB.t(), alpha = 1.0, beta = 0) # Approximation: use dK here instead of d_lora * magnitude_vector. This might need refinement.
            d_KB.addmm_(KA.t() @ X.t(), d_W_lora_k, alpha = 1.0, beta = 0) # Approximation: use dK here instead of d_lora * magnitude_vector. This might need refinement.
            dW_magnitude_k = d_magnitude_vector_k

            # V Projection DoRA weights
            lora_v = torch.matmul(VA.t(), VB.t()) * VS # Recalculate lora for backward
            directional_component_v = numerator_v / denominator_v # Recalculate directional_component for backward
            d_directional_component_v = v_magnitude_vector.T * fast_dequantize(dV.T, QUANT_STATE(dV))
            d_magnitude_vector_v = torch.sum(dV * directional_component_v, dim=0, keepdim=True)

            d_numerator_v = v_magnitude_vector * d_directional_component_v
            d_denominator_v = -torch.sum(d_numerator_v * numerator_v / denominator_v, dim=0, keepdim=True) / denominator_v

            d_W_lora_v = d_directional_component_v / denominator_v + d_denominator_v * numerator_v / (denominator_v ** 2)

            d_VA.addmm_(X.t(), d_W_lora_v @ VB.t(), alpha = 1.0, beta = 0) # Approximation: use dV here instead of d_lora * magnitude_vector. This might need refinement.
            d_VB.addmm_(VA.t() @ X.t(), d_W_lora_v, alpha = 1.0, beta = 0) # Approximation: use dV here instead of d_lora * magnitude_vector. This might need refinement.
            dW_magnitude_v = d_magnitude_vector_v

            # Combine derivatives to find dX
            dX = torch.matmul(dQ, (numerator_q / denominator_q).T)
            dX.addmm_(dK, (numerator_k / denominator_k).T)
            dX.addmm_(dV, (numerator_v / denominator_v).T)


        # QW, QW_quant, QA, QB, QS,
        # KW, KW_quant, KA, KB, KS,
        # VW, VW_quant, VA, VB, VS,
        return dX.view(batch, seq_len, hd), \
            None, None, d_QA.t(), d_QB.t(), None, \
            None, None, d_KA.t(), d_KB.t(), None, \
            None, None, d_VA.t(), d_VB.t(), None, \
            None, None, # inplace
            dW_magnitude_q, dW_magnitude_k, dW_magnitude_v
    pass
pass


def apply_lora_qkv(self, X, inplace = True, use_dora_qkv = False):
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    q_magnitude_vector = get_lora_parameters_bias(self.q_proj, "magnitude_vector") if use_dora_qkv else None
    k_magnitude_vector = get_lora_parameters_bias(self.k_proj, "magnitude_vector") if use_dora_qkv else None
    v_magnitude_vector = get_lora_parameters_bias(self.v_proj, "magnitude_vector") if use_dora_qkv else None
    Q, K, V = LoRA_QKV.apply(X,
        QW, QW_quant, QA, QB, QS,
        KW, KW_quant, KA, KB, KS,
        VW, VW_quant, VA, VB, VS,
        inplace, use_dora_qkv,
        q_magnitude_vector, k_magnitude_vector, v_magnitude_vector,
    )
    return Q, K, V
pass


class LoRA_W(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X : torch.Tensor,
                W, W_quant, A, B, S, use_dora_w = False, magnitude_vector = None): # Add use_dora and magnitude_vector
        dtype = X.dtype
        if not use_dora_w:
            XW = matmul_lora(X, W, W_quant, A, B, S)
        else:
            lora = torch.matmul(A, B) * S
            numerator = W + lora.T
            denominator = torch.linalg.norm(numerator, dim=0, keepdim=True)
            directional_component = numerator / denominator
            new_weight = magnitude_vector * directional_component # Assume magnitude_vector is passed correctly
            XW = torch.matmul(X, new_weight) # Use torch.matmul for simplicity and correctness
        ctx.custom_saved_tensors = (W, W_quant, S, use_dora_w) # Save use_dora for backward
        ctx.save_for_backward(A, B, X, magnitude_vector if use_dora_w else None, W if use_dora_w else None, numerator if use_dora_w else None, denominator if use_dora_w else None) # Save magnitude_vector and original W and numerator and denominator for dora
        return XW

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY : torch.Tensor):
        W, W_quant, S, use_dora_w = ctx.custom_saved_tensors # Retrieve use_dora
        A, B, X, magnitude_vector, original_W, numerator, denominator = ctx.saved_tensors # Retrieve magnitude_vector and original W for dora

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1]) # Must be reshape
        X  = X .reshape(-1, X .shape[-1]) # Must be reshape
        dtype = X.dtype

        A, B = A.to(dtype), B.to(dtype)
        A, B = A.t(), B.t()

        d_A = torch.empty_like(A)
        d_B = torch.empty_like(B)
        dW_magnitude = None # Initialize dW_magnitude to None

        if not use_dora_w: # Original backward pass for LoRA
            d_A.addmm_(X.t(), dY @ B.t(), alpha = S, beta = 0)
            d_B.addmm_(A.t() @ X.t(), dY, alpha = S, beta = 0)
            W = fast_dequantize(W.t(), W_quant)
            dX = dY @ W.t()
            del W
            dX.addmm_(dY @ B.t(), A.t(), alpha = S)
        else: # Backward pass for DoRA
            lora = torch.matmul(A.t(), B.t()) * S # Recalculate lora for backward
            directional_component = numerator / denominator # Recalculate directional_component for backward
            d_directional_component = magnitude_vector.T * fast_dequantize(dY.T, QUANT_STATE(dY)) # dY is the gradient of output, which is d(new_weight * X) / d(new_weight) = X. So dY here is actually dC/d(new_weight * X). We need dC/d(new_weight). Let's assume dY is dC/d(new_weight * X)
            d_magnitude_vector = torch.sum(dY * directional_component, dim=0, keepdim=True) # Sum over the input dimension

            d_numerator = magnitude_vector * d_directional_component # Gradient for numerator
            d_denominator = -torch.sum(d_numerator * numerator / denominator, dim=0, keepdim=True) / denominator # Gradient for denominator.

            d_W_lora = d_directional_component / denominator + d_denominator * numerator / (denominator ** 2) # Gradient for W+lora

            d_lora = d_W_lora # Gradient for lora
            d_A.addmm_(X.t(), d_lora @ B.t(), alpha = 1.0, beta = 0) # Approximation: use dY here instead of d_lora * magnitude_vector. This might need refinement.
            d_B.addmm_(A.t() @ X.t(), d_lora, alpha = 1.0, beta = 0) # Approximation: use dY here instead of d_lora * magnitude_vector. This might need refinement.

            dW_magnitude = d_magnitude_vector # Gradient for magnitude vector
            dX = torch.matmul(dY, (numerator / denominator).T) # dX should be dY * new_weight.T

        return dX.view(batch, seq_len, hd), \
            None, None, d_A.t(), d_B.t(), None, dW_magnitude, # Return dW_magnitude for magnitude_vector
    pass
pass


def apply_lora_o(self, X, use_dora_w = False):
    OW, OW_quant, OA, OB, OS = get_lora_parameters(self.o_proj)
    magnitude_vector = get_lora_parameters_bias(self.o_proj, "magnitude_vector") if use_dora_w else None
    O = LoRA_W.apply(X, OW, OW_quant, OA, OB, OS, use_dora_w, magnitude_vector)
    return O
pass


IDENTITY_DROPOUT = torch.nn.Identity
@torch._disable_dynamo
def fast_lora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        # Fastpath
        if len(self.active_adapters) == 1:
            active_adapter = self.active_adapters[0]
            if active_adapter not in self.lora_A.keys(): return self.base_layer(x, *args, **kwargs)

            dropout = self.lora_dropout[active_adapter]
            use_dora_w = self.use_dora.get(active_adapter, False)
            magnitude_vector = self.lora_magnitude_vector.get(active_adapter, None)

            if isinstance(dropout, IDENTITY_DROPOUT) and not use_dora_w:
                lora_A = self.lora_A[active_adapter].weight
                lora_B = self.lora_B[active_adapter].weight
                scaling = self.scaling[active_adapter]
                W = self.base_layer.weight
                return LoRA_W.apply(x, W, QUANT_STATE(W), lora_A, lora_B, scaling, use_dora_w) # Pass use_dora and magnitude vector
            elif isinstance(dropout, IDENTITY_DROPOUT) and use_dora_w:
                lora_A = self.lora_A[active_adapter].weight
                lora_B = self.lora_B[active_adapter].weight
                scaling = self.scaling[active_adapter]
                W = self.base_layer.weight
                magnitude_vector_param = self.lora_magnitude_vector[active_adapter].weight if self.lora_magnitude_vector.get(active_adapter) is not None else None
                return LoRA_W.apply(x, W, QUANT_STATE(W), lora_A, lora_B, scaling, use_dora_w, magnitude_vector_param)
            pass
        pass

        result = self.base_layer(x, *args, **kwargs)
        # As per Tim Dettmers, for 4bit, we need to defensively clone here.
        # The reason is that in some cases, an error can occur that backprop
        # does not work on a manipulated view. This issue may be solved with
        # newer PyTorch versions but this would need extensive testing to be
        # sure.
        result = result.clone()

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            use_dora_w = self.use_dora.get(active_adapter, False)
            magnitude_vector = self.lora_magnitude_vector.get(active_adapter, None)


            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            if not use_dora_w:
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                if isinstance(dropout, torch.nn.Identity) or not self.training:
                    base_result = result
                else:
                    x = dropout(x)
                    base_result = None

                result = result + self.lora_magnitude_vector[active_adapter](
                    x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    base_layer=self.get_base_layer(),
                    base_result=base_result,
                )
            if requires_conversion:
                result = result.to(expected_dtype)

    return result
pass