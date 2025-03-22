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
                gateW, gateW_quant, gateA, gateB, gateS, gate_m, use_gate_dora,
                  upW,   upW_quant, upA,   upB,   upS, up_m, use_up_dora,
                downW, downW_quant, downA, downB, downS, down_m, use_down_dora,
                _forward_function, _backward_function,
                inplace = True,):
        dtype = X.dtype

        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS, gate_m, use_gate_dora)
        g = matmul_lora(X,   upW,   upW_quant,   upA,   upB,   upS, up_m, use_up_dora)
        h = _forward_function(e, g)
        i = matmul_lora(h, downW, downW_quant, downA, downB, downS, down_m, use_down_dora)

        ctx.custom_saved_tensors = (
            gateW, gateW_quant, gateS, gate_m, use_gate_dora,
            upW, upW_quant, upS, up_m, use_up_dora,
            downW, downW_quant, downS, down_m, use_down_dora,
            _backward_function,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB,
                              X, e, g)
        ctx.inplace = inplace
        return i
    pass


    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY : torch.Tensor):
        gateW, gateW_quant, gateS, gate_m, use_gate_dora, upW, upW_quant, upS, up_m, use_up_dora, downW, downW_quant, downS, down_m, use_down_dora, \
            _backward_function = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, \
            X, e, g = ctx.saved_tensors

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

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS, down_m, use_down_dora)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA   = torch.empty_like(upA)
        d_upB   = torch.empty_like(upB)
        d_down_m = None if down_m is None else torch.zeros_like(down_m)
        d_up_m = None if up_m is None else torch.zeros_like(up_m)
        d_gate_m = None if gate_m is None else torch.zeros_like(gate_m)


        # Down projection LoRA weights
        d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha = downS, beta = 0)

        if use_down_dora:
            W_plus_AB = fast_dequantize(downW, downW_quant) + downS * (downA @ downB)
            V = W_plus_AB / W_plus_AB.norm(p=2, dim=0, keepdim=True)
            d_directional_component = dY.T @ h # (out_dim, in_dim)
            d_m_term = (d_directional_component * V).sum(dim=1, keepdim=True) # (out_dim, 1)
            d_V_term = down_m * (d_directional_component - (d_directional_component * V).sum(dim=1, keepdim=True) * V) # (out_dim, in_dim)
            d_lora = d_V_term # approximated, needs proper derivation for full DoRA
            if d_down_m is not None:
                d_down_m += d_m_term.sum() # scalar sum for magnitude


            d_downA_lora = d_lora @ downB.T
            d_downB_lora = downA.T @ d_lora

            d_downA.add_(d_downA_lora * downS)
            d_downB.add_(d_downB_lora * downS)


        # Up projection LoRA weights
        d_upA.addmm_(X.t(), df @ upB.t(), alpha = upS, beta = 0)
        d_upB.addmm_(upA.t() @ X.t(), df, alpha = upS, beta = 0)

        if use_up_dora:
            W_plus_AB = fast_dequantize(upW, upW_quant) + upS * (upA @ upB)
            V = W_plus_AB / W_plus_AB.norm(p=2, dim=0, keepdim=True)
            d_directional_component = df.T @ X # (out_dim, in_dim)
            d_m_term = (d_directional_component * V).sum(dim=1, keepdim=True) # (out_dim, 1)
            d_V_term = up_m * (d_directional_component - (d_directional_component * V).sum(dim=1, keepdim=True) * V) # (out_dim, in_dim)
            d_lora = d_V_term # approximated, needs proper derivation for full DoRA

            if d_up_m is not None:
                d_up_m += d_m_term.sum()

            d_upA_lora = d_lora @ upB.T
            d_upB_lora = upA.T @ d_lora

            d_upA.add_(d_upA_lora * upS)
            d_upB.add_(d_upB_lora * upS)


        # Gate projection LoRA weights
        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha = gateS, beta = 0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha = gateS, beta = 0)

        if use_gate_dora:
            W_plus_AB = fast_dequantize(gateW, gateW_quant) + gateS * (gateA @ gateB)
            V = W_plus_AB / W_plus_AB.norm(p=2, dim=0, keepdim=True)
            d_directional_component = de.T @ X # (out_dim, in_dim)
            d_m_term = (d_directional_component * V).sum(dim=1, keepdim=True) # (out_dim, 1)
            d_V_term = gate_m * (d_directional_component - (d_directional_component * V).sum(dim=1, keepdim=True) * V) # (out_dim, in_dim)
            d_lora = d_V_term # approximated, needs proper derivation for full DoRA

            if d_gate_m is not None:
                d_gate_m += d_m_term.sum()

            d_gateA_lora = d_lora @ gateB.T
            d_gateB_lora = gateA.T @ d_lora

            d_gateA.add_(d_gateA_lora * gateS)
            d_gateB.add_(d_gateB_lora * gateS)


        # dX  = matmul_lora(df, upW.t(), upW_quant, upB, upA, upS)
        # dX += matmul_lora(de, gateW.t(), gateW_quant, gateB, gateA, gateS)
        upW = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)
        del upW
        # dX += df @ upB.to(dtype).t() @ (upS * upA.to(dtype).t())
        dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)

        gateW = fast_dequantize(gateW.t(), gateW_quant)
        # dX += de @ gateW.t()
        dX.addmm_(de, gateW.t())
        del gateW
        # dX += de @ gateB.to(dtype).t() @ (gateS * gateA.to(dtype).t())
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha = gateS)

        # gateW, gateW_quant, gateA, gateB, gateS,
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        outputs = (dX.view(batch, seq_len, hd),
            None, None, d_gateA.t(), d_gateB.t(), d_gate_m, None,
            None, None,   d_upA.t(),   d_upB.t(), d_up_m, None,
            None, None, d_downA.t(), d_downB.t(), d_down_m, None,
            None, None, None, # _backward and _forward and inplace
        )
        return outputs
    pass
pass


from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
def apply_lora_mlp_swiglu(self, X, inplace = True, use_dora = False):
    gateW, gateW_quant, gateA, gateB, gateS, gate_m = get_lora_parameters(self.gate_proj, use_dora = use_dora)
    upW,     upW_quant,   upA,   upB,   upS, up_m = get_lora_parameters(self.  up_proj, use_dora = use_dora)
    downW, downW_quant, downA, downB, downS, down_m = get_lora_parameters(self.down_proj, use_dora = use_dora)
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS, gate_m, use_dora,
                         upW,     upW_quant, upA,   upB,   upS, up_m, use_dora,
                         downW, downW_quant, downA, downB, downS, down_m, use_dora,
                         swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel,
                         inplace,)
    return out
pass


from .geglu import geglu_exact_forward_kernel, geglu_exact_backward_kernel
def apply_lora_mlp_geglu_exact(self, X, inplace = True, use_dora = False):
    gateW, gateW_quant, gateA, gateB, gateS, gate_m = get_lora_parameters(self.gate_proj, use_dora = use_dora)
    upW,     upW_quant,   upA,   upB,   upS, up_m = get_lora_parameters(self.  up_proj, use_dora = use_dora)
    downW, downW_quant, downA, downB, downS, down_m = get_lora_parameters(self.down_proj, use_dora = use_dora)
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS, gate_m, use_dora,
                         upW,     upW_quant, upA,   upB,   upS, up_m, use_dora,
                         downW, downW_quant, downA, downB, downS, down_m, use_dora,
                         geglu_exact_forward_kernel, geglu_exact_backward_kernel,
                         inplace,)
    return out
pass


from .geglu import geglu_approx_forward_kernel, geglu_approx_backward_kernel
def apply_lora_mlp_geglu_approx(self, X, use_dora = False):
    gateW, gateW_quant, gateA, gateB, gateS, gate_m = get_lora_parameters(self.gate_proj, use_dora = use_dora)
    upW,     upW_quant,   upA,   upB,   upS, up_m = get_lora_parameters(self.  up_proj, use_dora = use_dora)
    downW, downW_quant, downA, downB, downS, down_m = get_lora_parameters(self.down_proj, use_dora = use_dora)
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS, gate_m, use_dora,
                         upW,     upW_quant, upA,   upB,   upS, up_m, use_dora,
                         downW, downW_quant, downA, downB, downS, down_m, use_dora,
                         geglu_approx_forward_kernel, geglu_approx_backward_kernel,)
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
                QW, QW_quant, QA, QB, QS, Qm, use_Q_dora,
                KW, KW_quant, KA, KB, KS, Km, use_K_dora,
                VW, VW_quant, VA, VB, VS, Vm, use_V_dora,
                inplace = True):
        dtype = X.dtype

        Q = matmul_lora(X, QW, QW_quant, QA, QB, QS, Qm, use_Q_dora)
        K = matmul_lora(X, KW, KW_quant, KA, KB, KS, Km, use_K_dora)
        V = matmul_lora(X, VW, VW_quant, VA, VB, VS, Vm, use_V_dora)

        ctx.custom_saved_tensors = (
            QW, QW_quant, QS, Qm, use_Q_dora,
            KW, KW_quant, KS, Km, use_K_dora,
            VW, VW_quant, VS, Vm, use_V_dora,
        )
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB,)
        ctx.inplace = inplace
        return Q, K, V
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, Qm, use_Q_dora, KW, KW_quant, KS, Km, use_K_dora, VW, VW_quant, VS, Vm, use_V_dora = \
            ctx.custom_saved_tensors
        X, QA, QB, KA, KB, VA, VB, = ctx.saved_tensors

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
        d_Qm = None if Qm is None else torch.zeros_like(Qm)
        d_Km = None if Km is None else torch.zeros_like(Km)
        d_Vm = None if Vm is None else torch.zeros_like(Vm)


        # Q Projection
        d_QA.addmm_(X.t(), dQ @ QB.t(), alpha = QS, beta = 0)
        d_QB.addmm_(QA.t() @ X.t(), dQ, alpha = QS, beta = 0)

        if use_Q_dora:
            W_plus_AB = fast_dequantize(QW, QW_quant) + QS * (QA @ QB)
            V = W_plus_AB / W_plus_AB.norm(p=2, dim=0, keepdim=True)
            d_directional_component = dQ.T @ X # (out_dim, in_dim)
            d_m_term = (d_directional_component * V).sum(dim=1, keepdim=True) # (out_dim, 1)
            d_V_term = Qm * (d_directional_component - (d_directional_component * V).sum(dim=1, keepdim=True) * V) # (out_dim, in_dim)
            d_lora = d_V_term # approximated, needs proper derivation for full DoRA
            if d_Qm is not None:
                d_Qm += d_m_term.sum()

            d_QA_lora = d_lora @ QB.T
            d_QB_lora = QA.T @ d_lora

            d_QA.add_(d_QA_lora * QS)
            d_QB.add_(d_QB_lora * QS)


        # K Projection
        d_KA.addmm_(X.t(), dK @ KB.t(), alpha = KS, beta = 0)
        d_KB.addmm_(KA.t() @ X.t(), dK, alpha = KS, beta = 0)

        if use_K_dora:
            W_plus_AB = fast_dequantize(KW, KW_quant) + KS * (KA @ KB)
            V = W_plus_AB / W_plus_AB.norm(p=2, dim=0, keepdim=True)
            d_directional_component = dK.T @ X # (out_dim, in_dim)
            d_m_term = (d_directional_component * V).sum(dim=1, keepdim=True) # (out_dim, 1)
            d_V_term = Km * (d_directional_component - (d_directional_component * V).sum(dim=1, keepdim=True) * V) # (out_dim, in_dim)
            d_lora = d_V_term # approximated, needs proper derivation for full DoRA
            if d_Km is not None:
                d_Km += d_m_term.sum()

            d_KA_lora = d_lora @ KB.T
            d_KB_lora = KA.T @ d_lora

            d_KA.add_(d_KA_lora * KS)
            d_KB.add_(d_KB_lora * KS)


        # V Projection
        d_VA.addmm_(X.t(), dV @ VB.t(), alpha = VS, beta = 0)
        d_VB.addmm_(VA.t() @ X.t(), dV, alpha = VS, beta = 0)

        if use_V_dora:
            W_plus_AB = fast_dequantize(VW, VW_quant) + VS * (VA @ VB)
            V = W_plus_AB / W_plus_AB.norm(p=2, dim=0, keepdim=True)
            d_directional_component = dV.T @ X # (out_dim, in_dim)
            d_m_term = (d_directional_component * V).sum(dim=1, keepdim=True) # (out_dim, 1)
            d_V_term = Vm * (d_directional_component - (d_directional_component * V).sum(dim=1, keepdim=True) * V) # (out_dim, in_dim)
            d_lora = d_V_term # approximated, needs proper derivation for full DoRA
            if d_Vm is not None:
                d_Vm += d_m_term.sum()

            d_VA_lora = d_lora @ VB.T
            d_VB_lora = VA.T @ d_lora

            d_VA.add_(d_VA_lora * VS)
            d_VB.add_(d_VB_lora * VS)


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

        # QW, QW_quant, QA, QB, QS,
        # KW, KW_quant, KA, KB, KS,
        # VW, VW_quant, VA, VB, VS,
        outputs = (dX.view(batch, seq_len, hd),
            None, None, d_QA.t(), d_QB.t(), d_Qm, None,
            None, None, d_KA.t(), d_KB.t(), d_Km, None,
            None, None, d_VA.t(), d_VB.t(), d_Vm, None,
            None,
        )
        return outputs
    pass
pass


def apply_lora_qkv(self, X, inplace = True, use_dora = False):
    QW, QW_quant, QA, QB, QS, Qm = get_lora_parameters(self.q_proj, use_dora = use_dora)
    KW, KW_quant, KA, KB, KS, Km = get_lora_parameters(self.k_proj, use_dora = use_dora)
    VW, VW_quant, VA, VB, VS, Vm = get_lora_parameters(self.v_proj, use_dora = use_dora)
    Q, K, V = LoRA_QKV.apply(X,
        QW, QW_quant, QA, QB, QS, Qm, use_dora,
        KW, KW_quant, KA, KB, KS, Km, use_dora,
        VW, VW_quant, VA, VB, VS, Vm, use_dora,
        inplace,
    )
    return Q, K, V
pass


class LoRA_W(torch.autograd.Function):
    """
    ### LoRA weights
    W = W + A @ B
    XW = X @ W = X @ W + X @ A @ B

    ### Backpropagation chain rule
    dC/dW = X.T @ dY

    ### LoRA weights
    dC/dA =       X.T @ dY @ B.T
    dC/dB = A.T @ X.T @ dY
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X : torch.Tensor,
                W, W_quant, A, B, S, m, use_dora):
        dtype = X.dtype
        if use_dora:
            W_plus_AB = fast_dequantize(W, W_quant) + S * (A @ B)
            V = W_plus_AB / W_plus_AB.norm(p=2, dim=0, keepdim=True)
            W_dora = m * V
            XW = torch.matmul(X, W_dora)
        else:
            XW = matmul_lora(X, W, W_quant, A, B, S, m, use_dora) # use_dora here is just passed for signature consistency

        ctx.custom_saved_tensors = (W, W_quant, S, m, use_dora)
        ctx.save_for_backward(A, B, X)
        return XW
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY : torch.Tensor):
        W, W_quant, S, m, use_dora = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1]) # Must be reshape
        X  = X .reshape(-1, X .shape[-1]) # Must be reshape
        dtype = X.dtype

        A, B = A.to(dtype), B.to(dtype)
        A, B = A.t(), B.t()

        d_A = torch.empty_like(A)
        d_B = torch.empty_like(B)
        d_m = None if m is None else torch.zeros_like(m)


        ### Weight projection LoRA weights
        # Weight projection
        d_A.addmm_(X.t(), dY @ B.t(), alpha = S, beta = 0)
        d_B.addmm_(A.t() @ X.t(), dY, alpha = S, beta = 0)


        if use_dora:
            W_plus_AB = fast_dequantize(W, W_quant) + S * (A @ B)
            V = W_plus_AB / W_plus_AB.norm(p=2, dim=0, keepdim=True) # Directional component
            d_directional_component = dY.T @ X # (out_dim, in_dim)
            d_m_term = (d_directional_component * V).sum(dim=1, keepdim=True) # (out_dim, 1)
            d_V_term = m * (d_directional_component - (d_directional_component * V).sum(dim=1, keepdim=True) * V) # (out_dim, in_dim)
            d_lora = d_V_term # approximated, needs proper derivation for full DoRA

            if d_m is not None:
                d_m += d_m_term.sum() # scalar sum for magnitude

            d_A_lora = d_lora @ B.T
            d_B_lora = A.T @ d_lora

            d_A.add_(d_A_lora * S)
            d_B.add_(d_B_lora * S)


        # Get derivative for dX
        W_dequant = fast_dequantize(W.t(), W_quant)
        dX = dY @ W_dequant
        del W_dequant
        if not use_dora: # LoRA case
            dX.addmm_(dY @ B.t(), A.t(), alpha = S) # dX += dY @ B.to(dtype).t() @ (S * A.to(dtype).t())
        else: # DoRA case - derivative of DoRA weight
            dX_dora = dY @ (m * V).T # derivative through DoRA weight
            dX += dX_dora
            del dX_dora

        # W, W_quant, A, B, S, m, use_dora
        return dX.view(batch, seq_len, hd), \
            None, None, d_A.t(), d_B.t(), None, d_m, None
    pass
pass


def apply_lora_o(self, X, use_dora = False):
    OW, OW_quant, OA, OB, OS, Om = get_lora_parameters(self.o_proj, use_dora = use_dora)
    O = LoRA_W.apply(X, OW, OW_quant, OA, OB, OS, Om, use_dora)
    return O
pass


IDENTITY_DROPOUT = torch.nn.Identity
@torch._disable_dynamo
def fast_lora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    raise NotImplementedError(
        "Unsloth: Currently not supported yet - reshaping done incorrectly"
    )
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
            if isinstance(dropout, IDENTITY_DROPOUT) and self.use_dora[active_adapter]: # check for DoRA
                lora_A = self.lora_A[active_adapter].weight
                lora_B = self.lora_B[active_adapter].weight
                scaling = self.scaling[active_adapter]
                W = self.base_layer.weight
                lora_m = self.lora_magnitude_vector[active_adapter] if self.use_dora[active_adapter] else None # Get magnitude parameter
                use_dora = self.use_dora[active_adapter] # Enable DoRA flag
                return LoRA_W.apply(x, W, QUANT_STATE(W), lora_A, lora_B, scaling, lora_m, use_dora) # Pass use_dora flag
            elif isinstance(dropout, IDENTITY_DROPOUT): # LoRA case
                lora_A = self.lora_A[active_adapter].weight
                lora_B = self.lora_B[active_adapter].weight
                scaling = self.scaling[active_adapter]
                W = self.base_layer.weight
                lora_m = None # No magnitude parameter for LoRA
                use_dora = False # Disable DoRA flag
                return LoRA_W.apply(x, W, QUANT_STATE(W), lora_A, lora_B, scaling, lora_m, use_dora) # Pass use_dora flag
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
            lora_m = self.lora_magnitude_vector[active_adapter] if self.use_dora[active_adapter] else None # Get magnitude parameter
            use_dora = self.use_dora[active_adapter] # Enable DoRA flag


            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            if not use_dora: # LoRA case
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else: # DoRA case
                if isinstance(dropout, torch.nn.Identity) or not self.training:
                    base_result = result
                else:
                    x = dropout(x)
                    base_result = None

                W = self.get_base_layer().weight # Get base layer weight
                result = result + self.lora_magnitude_vector[active_adapter](
                    x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    base_layer=self.get_base_layer(), # Pass base layer
                    base_result=base_result,
                    use_dora = use_dora, # Pass use_dora flag
                    lora_m = lora_m, # Pass magnitude parameter
                    W = W, # Pass base weight
                )
            if requires_conversion:
                result = result.to(expected_dtype)

    return result
pass