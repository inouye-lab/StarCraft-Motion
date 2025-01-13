
class TransformerModel(nn.Module):
    def __init__(self, seq_len, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_unit_type = nn.Embedding(512, 64, padding_idx=0)
        self.embed_owner = nn.Embedding(4, 3, padding_idx=0)
        self.embed_z_status = nn.Embedding(4, 3, padding_idx=0)
        self.embed_action = nn.Embedding(412, 64, padding_idx=0)
        self.attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        self.decoder = nn.Sequential(*[Block(hidden_dim) for _ in range(num_layers)])
        
        self.encode_mlp = MLP(573440, self.hidden_dim)    # encode MLP
        self.decode_mlp = MLP(self.hidden_dim, 8192)      # decode MLP

    def forward(self, tokens):
        # token size: B * T * N * A
        # B: batch size, T: sequence length, N: number of agents, A: number of attributes
        print(tokens.shape)
        embedding = torch.cat([
            self.embed_unit_type(tokens[:,:,:,0]),
            self.embed_owner(tokens[:,:,:,1]),
            self.embed_z_status(tokens[:,:,:,2]),
            self.embed_action(tokens[:,:,:,3]),
            tokens[:,:,:,4:10]
        ], dim=3)
        
        embedding = self.encode_mlp(embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2]*embedding.shape[3]))
        embedding = embedding + self.positional_encoding()

        output = self.decoder(embedding)
        continuous = self.mlp(output)
        return continuous