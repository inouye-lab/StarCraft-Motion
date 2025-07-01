import torch
import torch.nn as nn
import math
import copy
from torch.nn.parameter import Parameter
import einops

class Tokenizer:
    def __init__(self, eos_token=-1, pad_token=0):
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.action_mapping = torch.load('utils/action_mapping.pt')

    def __call__(self, data):
        return self.tokenize(data)
    # call method
    
    def tokenize(self, data):
        # todo: filter out the agents that are never moved in the whole sequence.
        
        # data format: (T, N, A) where T is the number of frames, N is the number of agents, and A is the number of attributes; A is 15 dimensional:  unit.unit_type, unit.owner, unit.pos.x, unit.pos.y, unit.pos.z (21.25*raw_z - 85), unit_z_status, unit.facing (255/(2*pi)), unit.health (health/health_max * 255), unit.build_progress(raw*255), unit.last_action_raw, unit.last_action_type, unit.pos_of_last_action.x, unit.pos_of_last_action.y, target_tag_high_digits, target_tag_low_digits
        # categorical: 0, 1, 5, 9, 10 (9,10 together encodes action), 13,14 (together encode target id.)
        
        # continuous: 2,3 (relative), 7, 8, 11, 12 (relative)
        # dropping: 4, 6
        # adding: (target id).
        # Remove all the non-existence objects.
        # We need add feature of unit id explicit.
        assert data.ndim == 4, "Data must have 4 dimensions: (B, T, N, A)"
        assert data.shape[3] == 18, "Data must have 18 attributes"
    
        # prepare the reference point for relative encoding.

        reference = torch.randint(0, 256, (data.size(0), 1, 1, 2), device=data.device)
        # Attr 0: Unit type. Unit type depends on whether owner==16. If owner==16, then we need to map this unit type to a new id (+255). Then, we encode this id using the embedding layer.
        # Attr 1: Unit owner. We need to encode this using the embedding layer (3 dimension).
        # Attr 5: Unit z_status. We need to encode this using the embedding layer (3 dimension).
        # Attr 9-10: Together, they encode the action. We need to encode this using the embedding layer (xx dimension).
        token = torch.empty(data.shape[0], data.shape[1], data.shape[2], 9, dtype=torch.long, device=data.device)

        token[:,:,:,0] = data[:,:,:,0] + 256 * (data[:,:,:,1] == 16)
        token[:,:,:,1] = (data[:,:,:,1] == 16) * 3 + (data[:,:,:,1] != 16) * data[:,:,:,1]
        # token[:,:,:,2] = data[:,:,:,5]
        # token[:,:,:,3] = self.action_mapping[data[:,:,:,9].to(int), data[:,:,:,10].to(int)] 

        # Pass them to the embedding layer.
        ## We need to add the positional encoding.
        

        # Then we go to the continuous attributes. 
        # token[:,:,:,4:6] = self.relative_encode(data[:,:,:,2:4], reference)
        # token[:,:,:,6] = data[:,:,:,7]
        # token[:,:,:,7] = data[:,:,:,8]
        # token[:,:,:,8:10] = self.relative_encode(data[:,:,:, 11:13], reference)
        token[:,:,:,2] = data[:,:,:,5]
        token[:,:,:,3:5] = self.relative_encode(data[:,:,:,2:4], reference)
       
        token[:,:,:,5] = data[:,:,:,7]
        token[:,:,:,6] = data[:,:,:,8]
        token[:,:,:,7] = data[:,:,:,9]
        token[:,:,:,8] = data[:,:,:,10]
        
        
        # make it incremental
        incremental_token = torch.cat([token[:,1:,:,0:3], token[:, 1:, :, 3:]-token[:, :-1, :, 3:]], dim=3)
        
        token = torch.cat([token[:,0,:,:].unsqueeze(1), incremental_token], dim=1)

        
        # Then deal with the empty tokens.
        padding_mask = torch.linalg.norm(data, dim=3) == 0

        # Transfer to incremental token.

        # check and return
        assert token.ndim == 4 # output should be 3D tensor. (B, T, N, A) NA together should be a long token.
        return token, padding_mask
 
    def relative_encode(self, seg, reference):
        # We first choose a random point in the region (0,0) to (255, 255) as the reference point. Then each point is encoded as the relative position to this reference point. Then we normalize it to (-1, 1). The encoding is different inside the same batch.
        seg = seg - reference
        return seg


class MLP(nn.Module):
    def __init__(self, d_in, d_out):  # in MLP: n_state (4 * n_embd)
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(d_in, d_in * 2)
        self.c_proj = nn.Linear(d_in * 2, d_out)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class SpatialBlock(nn.Module):
    # SpatialBlock: TransformerEncoderLayer with causal_mask=False, T collapes to the batch, thus ignore temporal information.
    def __init__(self, n_embd, n_heads):
        super(SpatialBlock, self).__init__()
        self.decoder = nn.TransformerEncoderLayer(d_model=n_embd, 
                                                  nhead=n_heads, 
                                                  dim_feedforward=1024, 
                                                  dropout=0.1, 
                                                  activation='relu', 
                                                  layer_norm_eps=1e-05, 
                                                  batch_first=True)
    
    def forward(self, x, padding_mask=None):
        b, t, n, a = x.shape
        if padding_mask is not None:
            padding_mask = einops.rearrange(padding_mask, 'B T N -> (B T) N')
        idx_nouse = (padding_mask.sum(dim=1)==9).nonzero().squeeze()
        idx_use = (padding_mask.sum(dim=1)!=9).nonzero().squeeze()
        bs = idx_use.shape[0]
        x = einops.rearrange(x, 'B T N A -> (B T) N A')
        y = self.decoder(x[idx_use], is_causal=False, src_key_padding_mask=padding_mask[idx_use])
        new_x = torch.zeros_like(x, device=x.device)
        new_x[idx_use] = y
        new_x[idx_nouse] = x[idx_nouse]
        x = einops.rearrange(new_x, '(B T) N A -> B T N A', B=b)
        return x


class TemporalBlock(SpatialBlock):
    # TemporalBlock: TransformerEncoderLayer with causal_mask=True, B collapes to the batch, thus ignore spatial information.
    def forward(self, x, padding_mask=None):
        b, t, n, a = x.shape
        if padding_mask is not None:
            padding_mask = einops.rearrange(padding_mask, 'B T N -> (B N) T')
        idx_nouse = (padding_mask.sum(dim=1)==9).nonzero().squeeze()
        idx_use = (padding_mask.sum(dim=1)!=9).nonzero().squeeze()
        x = einops.rearrange(x, 'B T N A -> (B N) T A')
        y = self.decoder(x[idx_use], is_causal=True, src_key_padding_mask=padding_mask[idx_use])
        new_x = torch.zeros_like(x, device=x.device)
        new_x[idx_use] = y
        new_x[idx_nouse] = x[idx_nouse]
        x = einops.rearrange(new_x, '(B N) T A -> B T N A ', B=b)
        return x
        

class TransformerModel(nn.Module):
    def __init__(self, n_embd, n_ctx, n_heads, n_layer, temporal=True, spatial=True):
        super(TransformerModel, self).__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd       # hidden dimension
        self.n_heads = n_heads
        self.n_ctx = n_ctx         # number of frames
        self.embed_unit_type = nn.Embedding(512, 494, padding_idx=0)
        self.embed_owner = nn.Embedding(4, 6, padding_idx=0)
        self.embed_z_status = nn.Embedding(6, 6, padding_idx=0)
        # self.embed_action = nn.Embedding(412, 64, padding_idx=0)
        # self.attention_mask = torch.tril(torch.ones(n_ctx, n_ctx)).unsqueeze(0)
        self.encode_mlp = MLP(512, n_embd)    # encode MLP # todo
        self.decode_mlp = MLP(n_embd, 4)      # decode MLP
        self.decode_mlp_intention = MLP(n_embd, 26)      # decode MLP
        self.positional_embedding = SinusoidalPositionalEncoding(n_ctx, n_embd)
        if temporal and spatial:
            self.h = nn.ModuleList([SpatialBlock(n_embd, n_heads) if i % 2 else TemporalBlock(n_embd, n_heads) for i in range(n_layer)])
        elif spatial:
            self.h = nn.ModuleList([SpatialBlock(n_embd, n_heads) for _ in range(n_layer)])
        elif temporal:
            self.h = nn.ModuleList([TemporalBlock(n_embd, n_heads)for _ in range(n_layer)])
        else:
            raise NotImplementedError
            
    def forward(self, input_ids, padding_mask=None):
        # input_ids: (B,T-1,N,a)
        embedding = torch.cat([
            self.embed_unit_type(input_ids[:,:,:,0]),
            self.embed_owner(input_ids[:,:,:,1]),
            self.embed_z_status(input_ids[:,:,:,2]),
            input_ids[:,:,:,3:9] / 255.0
        ], dim=3)  
        # embedding: (B,T-1,N,64+3+3+64+6=140)
        # First way: B*N,T-1, a put the agents to batch dimension. huge approximation. (no spatial information)
        # Second way: B*(T-1), N (several hundreds in a ten-second window), a (no temporal information)
        # Fourth way: alternative first and second ways.
        embedding = self.encode_mlp(embedding) # embedding: (B*(T-1), N, n_embd)

        hidden_states = self.positional_embedding(embedding)

        for block in self.h:
            hidden_states = block(hidden_states, padding_mask)
        output_continuous, output_intention = self.decode_mlp(hidden_states), self.decode_mlp_intention(hidden_states)
        return output_continuous, output_intention


    # def positional_encoding(self, x):
    #     B,T,N,A = x.shape
    #     if self.n_embd % 2 != 0:
    #         raise ValueError("Cannot use sin/cos positional encoding with "
    #                         "odd dim (got dim={:d})".format(self.n_embd))
    #     pe = torch.zeros(1, T, N, self.n_embd)
    #     position = torch.arange(0, self.n_ctx).unsqueeze(1)
    #     div_term = torch.exp((torch.arange(0, self.n_ctx, 2, dtype=torch.float) *
    #                         -(math.log(10000.0) / self.n_embd)))
    #     pe[0, 0::2, :, :] = torch.sin(position.float() * div_term)
    #     pe[0, 1::2, :, :] = torch.cos(position.float() * div_term)
    #     return pe

    def save_model(self, optimizer, path, epoch=None, loss=None):
        """Save the model state, optimizer state, epoch, and loss to a file.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer associated with the model.
            path (str): Path where the model will be saved.
            epoch (int, optional): Current epoch number. Defaults to None.
            loss (float, optional): Current loss. Defaults to None.
        """
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        if epoch is not None:
            state['epoch'] = epoch
        if loss is not None:
            state['loss'] = loss
        
        torch.save(state, path)
        # print(f"Model saved to {path}")

    def load_model(self, optimizer, path, device='cpu'):
        """Load the model state, optimizer state, and other training information.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer associated with the model.
            path (str): Path from where the model will be loaded.
            device (str, optional): Device to load the model onto ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            tuple: Loaded epoch and loss if they were saved, otherwise None for each.
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)
        print(f"Model loaded from {path}, Epoch: {epoch}, Loss: {loss}")
        
        return epoch, loss


class SimpleMLP(nn.Module):
    def __init__(self, n_embd):
        super(SimpleMLP, self).__init__()
        self.n_embd = n_embd       # hidden dimension
        self.embed_unit_type = nn.Embedding(512, 494, padding_idx=0)
        self.embed_owner = nn.Embedding(4, 6, padding_idx=0)
        self.embed_z_status = nn.Embedding(6, 6, padding_idx=0)
        self.encode_mlp = MLP(512, n_embd)    # encode MLP # todo
        self.decode_mlp = MLP(n_embd, 4)      # decode MLP
        self.decode_mlp_intention = MLP(n_embd, 26)      # decode MLP
            
    def forward(self, input_ids, padding_mask=None):
        # input_ids: (B,T-1,N,a)
        embedding = torch.cat([
            self.embed_unit_type(input_ids[:,:,:,0]),
            self.embed_owner(input_ids[:,:,:,1]),
            self.embed_z_status(input_ids[:,:,:,2]),
            input_ids[:,:,:,3:9] / 255.0
        ], dim=3)  # embedding: (B,T-1,N,64+3+3+64+6=140)
        # First way: B*N,T-1, a put the agents to batch dimension. huge approximation. (no spatial information)
        # Second way: B*(T-1), N (several hundreds in a ten-second window), a (no temporal information)
        # Fourth way: alternative first and second ways.
        hidden_states = self.encode_mlp(embedding) # embedding: (B*(T-1), N, n_embd)
        # one way:
        output_continuous, output_intention = self.decode_mlp(hidden_states), self.decode_mlp_intention(hidden_states)
        return output_continuous, output_intention


    def save_model(self, optimizer, path, epoch=None, loss=None):
        """Save the model state, optimizer state, epoch, and loss to a file.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer associated with the model.
            path (str): Path where the model will be saved.
            epoch (int, optional): Current epoch number. Defaults to None.
            loss (float, optional): Current loss. Defaults to None.
        """
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        if epoch is not None:
            state['epoch'] = epoch
        if loss is not None:
            state['loss'] = loss
        
        torch.save(state, path)
        # print(f"Model saved to {path}")

    def load_model(self, optimizer, path, device='cpu'):
        """Load the model state, optimizer state, and other training information.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer associated with the model.
            path (str): Path from where the model will be loaded.
            device (str, optional): Device to load the model onto ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            tuple: Loaded epoch and loss if they were saved, otherwise None for each.
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)
        print(f"Model loaded from {path}, Epoch: {epoch}, Loss: {loss}")
        
        return epoch, loss


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learnable) sinusoidal positional encoding.
    Given an input tensor of shape [B, T, N, E], it adds a   
    positional signal of shape [1, T, 1, E] where:
        PE[pos, 2i]   =  sin(pos / 10000^(2i/E))
        PE[pos, 2i+1] =  cos(pos / 10000^(2i/E))
    """
    def __init__(self, n_ctx: int, n_embd: int):
        super().__init__()
        # create constant positional encodings matrix [n_ctx, n_embd]
        pe = torch.zeros(n_ctx, n_embd)
        position = torch.arange(0, n_ctx, dtype=torch.float).unsqueeze(1)  # [n_ctx, 1]
        div_term = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float) *
                             -(math.log(10000.0) / n_embd))               # [n_embd/2]
        pe[:, 0::2] = torch.sin(position * div_term)                       # even dims
        pe[:, 1::2] = torch.cos(position * div_term)                       # odd dims

        # register as buffer so it moves with the model but isn't a parameter
        self.register_buffer('positional_encoding', pe)  # [n_ctx, n_embd]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, N, E], where E == n_embd and T <= n_ctx
        Returns:
            x + positional encoding of shape [B, T, N, E]
        """
        B, T, N, E = x.shape
        # slice out the first T positions and reshape to [1, T, 1, E]
        pe_T = self.positional_encoding[:T]         # [T, E]
        print(pe_T)
        pe_T = pe_T.unsqueeze(0).unsqueeze(2)       # [1, T, 1, E]
        return x + pe_T