import torch
from torch import nn

D = Hidden_size = 768
MLP_size = 3072
Heads = 12
image_size = 224
batch_size = 4096
patch_size = 16
epoch_size = 10
classes = ["pizza", "steak", "sushi"]


class Net(nn.Module):
    def __init__(self, image_size, patch_size, drop_rate=0.1):
        super().__init__()
        self.image_size = image_size  # 224
        self.patch_size = patch_size  # 16
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=D,
                               kernel_size=patch_size,
                               stride=patch_size)
        self.flatten1 = nn.Flatten(2)
        batch_size = 75
        N = num_of_patches = int((image_size / patch_size) ** 2)
        self.class_token = nn.Parameter(torch.randn(batch_size, 1, D), requires_grad=True)  # [75, 1, 768]
        self.position_embedding = nn.Parameter(torch.randn(batch_size, num_of_patches + 1, D),
                                               requires_grad=True)  # [75, 197, 768]

        self.transformer_encoder = TransformerEncoder(drop_rate)

        self.clissification_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, len(classes))
        )

    def forward(self, input):  # [75, 3, 224, 224]
        # Linear Projection of Flattened Patches
        conv1 = self.conv1(input)  # [75, 768, 14, 14]
        patch_embedding = self.flatten1(conv1).permute(0, 2, 1)  # [75, 196, 768]
        patch_embedding = torch.cat((self.class_token, patch_embedding), dim=1)  # [75, 197, 768]
        patch_and_position_embedding = patch_embedding + self.position_embedding  # [75, 197, 768]

        # Transformer Encoder
        encoder_output = self.transformer_encoder(patch_and_position_embedding)  # [75, 197, 768]

        # Classification Head
        output = self.clissification_head(encoder_output[:, 0])  # [75, 3]
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.LN = nn.LayerNorm(D)
        self.MSA = MSA(drop_rate)
        self.MLP = MLP()

    def forward(self, input):
        norm1 = self.LN(input)  # [75, 197, 768]

        # MSA
        msa = self.MSA(norm1)  # [75, 197, 768]

        add1 = msa + input  # [75, 197, 768]

        norm2 = self.LN(add1)  # [75, 197, 768]

        # MLP
        mlp = self.MLP(norm2)  # [75, 197, 768]

        add2 = mlp + add1  # [75, 197, 768]
        output = add2  # [75, 197, 768]
        return output


class MSA(nn.Module):
    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_rate)

        self.output = nn.Linear(D, D)

    def forward(self, input):
        batch_size, seq_length, embed_dim = input.size()

        q = self.q(input)
        k = self.k(input)
        v = self.v(input)

        q = q.view(batch_size, seq_length, Heads, embed_dim // Heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_length, Heads, embed_dim // Heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_length, Heads, embed_dim // Heads).permute(0, 2, 1, 3)

        attention = torch.matmul(q, k.permute(0, 1, 3, 2)) / (D ** 0.5)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        output = torch.matmul(attention, v)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
        output = self.output(output)
        return output


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D, MLP_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(MLP_size, D),
            nn.Dropout(0.1)
        )

    def forward(self, input):
        output = self.mlp(input)
        return output
