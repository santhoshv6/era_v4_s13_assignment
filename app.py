"""
SmolLM2-135M Gradio Deployment App
For Hugging Face Spaces
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os


# ============================================================================
# Model Implementation (same as training notebook)
# ============================================================================

@dataclass
class SmolLM2Config:
    """SmolLM2-135M configuration"""
    block_size: int = 256
    vocab_size: int = 100  # Will be loaded from checkpoint
    n_layer: int = 30
    n_head: int = 9
    n_kv_head: int = 3
    n_embd: int = 576
    intermediate_size: int = 1536
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        assert self.n_head % self.n_kv_head == 0
        self.n_query_groups = self.n_head // self.n_kv_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # FIX: shape to [1, 1, seq_len, head_dim] for broadcasting
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_query_groups = config.n_query_groups
        
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.block_size,
            theta=config.rope_theta
        )
        
        self.o_proj.SMOLLM_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        k = k.repeat_interleave(self.n_query_groups, dim=1)
        v = v.repeat_interleave(self.n_query_groups, dim=1)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.down_proj.SMOLLM_SCALE_INIT = 1

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SmolLM2(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SMOLLM_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.model.embed_tokens(idx)
        
        for block in self.model.layers:
            x = block(x)
        
        x = self.model.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Gradio Interface
# ============================================================================

class ModelInference:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.stoi = None
        self.itos = None
        self.loaded = False
        
    def load_model(self, checkpoint_path="checkpoints/model_deployment.pt"):
        """Load trained model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            return False, "Checkpoint not found. Please train the model first."
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            config = checkpoint['model_config']
            
            # Create model
            self.model = SmolLM2(config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load vocabulary
            if 'vocab' in checkpoint:
                # Vocab saved in checkpoint
                self.stoi = checkpoint['vocab']['stoi']
                self.itos = checkpoint['vocab']['itos']
            else:
                # Reconstruct vocab from training data (same as training)
                # Try to load from input-1.txt
                vocab_file = "input-1.txt"
                if os.path.exists(vocab_file):
                    with open(vocab_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    chars = sorted(list(set(text)))
                    self.stoi = {ch: i for i, ch in enumerate(chars)}
                    self.itos = {i: ch for i, ch in enumerate(chars)}
                    print(f"✓ Vocabulary reconstructed from {vocab_file}: {len(chars)} characters")
                else:
                    # Last resort: use printable ASCII
                    chars = [chr(i) for i in range(32, 127)]
                    self.stoi = {ch: i for i, ch in enumerate(chars)}
                    self.itos = {i: ch for i, ch in enumerate(chars)}
                    print(f"⚠ Warning: Using fallback ASCII vocabulary ({len(chars)} chars)")
            
            vocab_size = len(self.stoi)
            self.loaded = True
            return True, f"Model loaded successfully! Device: {self.device}, Vocab size: {vocab_size}"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def encode(self, text):
        """Encode text to tokens"""
        return [self.stoi.get(ch, 0) for ch in text]
    
    def decode(self, tokens):
        """Decode tokens to text"""
        return ''.join([self.itos.get(t, '?') for t in tokens])
    
    def generate_text(self, prompt, max_tokens=200, temperature=0.8, top_k=40):
        """Generate text from prompt"""
        if not self.loaded:
            return "Model not loaded. Please load checkpoint first."
        
        if not prompt:
            prompt = "\n"
        
        # Encode prompt
        tokens = self.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        # Decode
        generated_text = self.decode(output_ids[0].cpu().tolist())
        return generated_text


# Initialize inference
inference = ModelInference()


def generate_interface(prompt, max_tokens, temperature, top_k, load_model_btn):
    """Gradio generate function"""
    if load_model_btn and not inference.loaded:
        success, msg = inference.load_model()
        if not success:
            return msg
    
    result = inference.generate_text(prompt, int(max_tokens), temperature, int(top_k))
    return result


def load_model_click():
    """Load model button handler"""
    success, msg = inference.load_model()
    return msg


# Create Gradio interface
with gr.Blocks(title="SmolLM2-135M Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 SmolLM2-135M Text Generation
    
    This is a SmolLM2-135M model trained from scratch with Grouped-Query Attention (GQA).
    
    ## Model Specifications
    - **Parameters**: ~135M
    - **Architecture**: 30 layers, 9 attention heads, 3 KV heads (GQA)
    - **Training**: 5050 steps with checkpointing
    - **Optimizations**: torch.compile, bfloat16, flash attention
    """)
    
    with gr.Row():
        with gr.Column():
            load_btn = gr.Button("🔄 Load Model", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            gr.Markdown("""
            ### Architecture Highlights
            - ✅ Grouped-Query Attention (GQA)
            - ✅ RMSNorm (faster than LayerNorm)
            - ✅ SwiGLU activation
            - ✅ Rotary Position Embeddings (RoPE)
            """)
    
    gr.Markdown("## Generate Text")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=3
            )
            
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Max Tokens"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=40,
                    step=1,
                    label="Top-K"
                )
            
            generate_btn = gr.Button("✨ Generate", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Generated Text",
                lines=15,
                interactive=False
            )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["ROMEO:\n", 200, 0.8, 40],
            ["KING RICHARD II:\n", 200, 0.8, 40],
            ["To be or not to be", 150, 0.7, 30],
            ["Once upon a time", 200, 0.9, 50],
            ["JULIET:\n", 200, 0.8, 40],
            ["The winter of our discontent", 180, 0.8, 40],
        ],
        inputs=[prompt, max_tokens, temperature, top_k],
        label="📝 Try These Example Prompts",
    )
    
    gr.Markdown("""
    ## About This Model
    
    This model was trained as part of Session 13 assignment, demonstrating:
    - Reverse-engineering SmolLM2 architecture from HuggingFace config
    - Implementing Grouped-Query Attention for efficiency  
    - Proper checkpoint save/resume (trained 5000 steps, saved, then resumed for 50 more)
    - State-of-the-art optimizations (flash attention, bfloat16, torch.compile)
    
    **Note**: This is a demonstration model trained for only 5050 steps. 
    The architecture is production-ready, but the model needs more training for high-quality outputs.
    
    ### Links
    - [GitHub Repository](#)
    - [Training Notebook](#)
    - [Original SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
    """)
    
    # Event handlers
    load_btn.click(
        fn=load_model_click,
        inputs=[],
        outputs=[load_status]
    )
    
    generate_btn.click(
        fn=lambda p, m, t, k: inference.generate_text(p, int(m), t, int(k)),
        inputs=[prompt, max_tokens, temperature, top_k],
        outputs=[output]
    )


if __name__ == "__main__":
    # Auto-load model on startup
    print("Starting SmolLM2-135M Demo...")
    success, msg = inference.load_model()
    print(msg)
    
    # Launch Gradio
    demo.launch(share=False)
