import numpy as np

class SimplifiedTransformer:
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        """
        Initialize a simplified transformer model.
        
        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        
        # Initialize weights (simplified)
        self.head_dim = d_model // nhead
        
        # Multi-head attention weights
        self.wq = np.random.randn(d_model, d_model) * 0.1
        self.wk = np.random.randn(d_model, d_model) * 0.1
        self.wv = np.random.randn(d_model, d_model) * 0.1
        self.wo = np.random.randn(d_model, d_model) * 0.1
        
        # Feed-forward weights
        self.w1 = np.random.randn(d_model, dim_feedforward) * 0.1
        self.w2 = np.random.randn(dim_feedforward, d_model) * 0.1
        
        # Layer normalization parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)
    
    def layer_norm(self, x, gamma, beta, eps=1e-6):
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta
    
    def dropout(self, x, training=True):
        """Apply dropout."""
        if not training or self.dropout_rate == 0:
            return x
        mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
        return x * mask
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Optional mask
            
        Returns:
            attention_output, attention_weights
        """
        # Compute attention scores
        matmul_qk = np.matmul(q, k.transpose(0, 1, 3, 2))

        
        # Scale attention scores
        dk = np.sqrt(self.head_dim)
        scaled_attention_logits = matmul_qk / dk
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scaled_attention_logits)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute output
        output = np.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def softmax(self, x):
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.reshape(batch_size, -1, self.nhead, self.head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x, batch_size):
        """Combine the heads back."""
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, -1, self.d_model)
    
    def feed_forward(self, x):
        """Apply feed-forward network."""
        hidden = np.maximum(0, np.matmul(x, self.w1))  # ReLU
        hidden = self.dropout(hidden)
        output = np.matmul(hidden, self.w2)
        return output
    
    def multi_head_attention(self, x, mask=None):
        """Apply multi-head attention."""
        batch_size = x.shape[0]
        
        # Linear projections
        q = np.matmul(x, self.wq)  # (batch_size, seq_len, d_model)
        k = np.matmul(x, self.wk)  # (batch_size, seq_len, d_model)
        v = np.matmul(x, self.wv)  # (batch_size, seq_len, d_model)
        
        # Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, nhead, seq_len, head_dim)
        k = self.split_heads(k, batch_size)  # (batch_size, nhead, seq_len, head_dim)
        v = self.split_heads(v, batch_size)  # (batch_size, nhead, seq_len, head_dim)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Combine heads
        attention_output = self.combine_heads(attention_output, batch_size)
        
        # Final linear projection
        output = np.matmul(attention_output, self.wo)
        
        return output, attention_weights
    
    def forward(self, x, mask=None):
        """Forward pass through the transformer layer."""
        # Multi-head attention
        attn_output, attention_weights = self.multi_head_attention(x, mask)
        attn_output = self.dropout(attn_output)
        
        # Add & Norm
        out1 = x + attn_output
        out1 = self.layer_norm(out1, self.ln1_gamma, self.ln1_beta)
        
        # Feed-forward
        ffn_output = self.feed_forward(out1)
        ffn_output = self.dropout(ffn_output)
        
        # Add & Norm
        out2 = out1 + ffn_output
        out2 = self.layer_norm(out2, self.ln2_gamma, self.ln2_beta)
        
        return out2, attention_weights
    
    def get_visualization_data(self, input_seq):
        """
        Process input sequence and return data for visualization.
        
        Args:
            input_seq: Input sequence (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary with visualization data
        """
        # Ensure input is numpy array with correct shape
        if isinstance(input_seq, list):
            # Convert list of tokens to a simple embedding (one-hot)
            seq_len = len(input_seq)
            batch_size = 1
            # Create a simple embedding (random for demonstration)
            input_embed = np.random.randn(batch_size, seq_len, self.d_model) * 0.1
        else:
            batch_size, seq_len, _ = input_seq.shape
            input_embed = input_seq
        
        # Forward pass
        output, attention_weights = self.forward(input_embed)
        
        # Prepare visualization data
        vis_data = {
            "input_tokens": input_seq if isinstance(input_seq, list) else ["token_" + str(i) for i in range(seq_len)],
            "attention_weights": attention_weights.tolist(),
            "model_dimensions": {
                "d_model": self.d_model,
                "nhead": self.nhead,
                "head_dim": self.head_dim,
                "dim_feedforward": self.dim_feedforward
            }
        }
        
        return vis_data

# Example usage
if __name__ == "__main__":
    # Create a transformer layer
    transformer = SimplifiedTransformer(d_model=512, nhead=8)
    
    # Create a sample input
    batch_size = 1
    seq_len = 5
    x = np.random.randn(batch_size, seq_len, 512)
    
    # Forward pass
    output, attention_weights = transformer.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")