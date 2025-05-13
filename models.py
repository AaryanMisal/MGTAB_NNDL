import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import HGTConv, RGCNConv
from layer import RGTLayer, SimpleHGNLayer
import torch.nn.functional as F
from torchvision import models

class RGT(nn.Module):
    def __init__(self, args):
        super(RGT, self).__init__()

        self.linear1 = nn.Linear(args.features_num, args.hidden_dimension)
        self.RGT_layer1 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension,
                                   trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        # self.RGT_layer2 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out1 = torch.nn.Linear(args.hidden_dimension, args.out_channel)
        self.out2 = torch.nn.Linear(args.out_channel, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, features, edge_index, edge_type):

        user_features = self.drop(self.ReLU(self.linear1(features)))
        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        user_features = self.drop(self.ReLU(self.out1(user_features)))
        x = self.out2(user_features)

        return x

class CrossModalAttention(nn.Module):
    def __init__(self, dim_node, dim_img, heads=4, dropout=0.1):
        super().__init__()
        self.proj_q = nn.Linear(dim_node, dim_node)
        self.proj_k = nn.Linear(dim_img, dim_node)
        self.proj_v = nn.Linear(dim_img, dim_node)
        self.attn  = nn.MultiheadAttention(embed_dim=dim_node, num_heads=heads, dropout=dropout)
        self.norm  = nn.LayerNorm(dim_node)
        self.ff    = nn.Sequential(nn.Linear(dim_node, dim_node),nn.ReLU(), nn.Linear(dim_node, dim_node),)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feats, img_feats, mask=None):
        x1 = self.norm(node_feats)
        Q = self.proj_q(x1).unsqueeze(1)   
        K = self.proj_k(img_feats).unsqueeze(1) 
        V = self.proj_v(img_feats).unsqueeze(1)

        # MultiheadAttention 
        attn_out, _ = self.attn(Q, K, V, key_padding_mask=mask)
        attn_out = attn_out.squeeze(1) 

        out = node_feats + self.dropout(attn_out)
        out = out + self.dropout(self.ff(self.norm(out)))
        return out

class GatedFusion(nn.Module):
    def __init__(self, dim_node, dim_img):
        super().__init__()
        self.fc = nn.Linear(dim_node + dim_img, 1)
        nn.init.zeros_(self.fc.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_feats, img_feats):
        concat = torch.cat([node_feats, img_feats], dim=1) 
        alpha = self.sigmoid(self.fc(concat))             
        fused = alpha * node_feats + (1 - alpha) * img_feats
        return fused, alpha

class RGT_multimodal(nn.Module):
    def __init__(self, args):
        super(RGT_multimodal, self).__init__()
        self.dropout = args.dropout
        # Text branch projection
        self.text_proj = nn.Sequential(nn.Linear(args.features_num, args.hidden_dimension), nn.LeakyReLU(), nn.Dropout(self.dropout))
        # Image branch projection
        self.img_proj = nn.Sequential(nn.Linear(args.img_feat_dim, args.hidden_dimension), nn.LeakyReLU(), nn.Dropout(self.dropout))
        # Cross-modal attention blocks
        self.cross_node2img = CrossModalAttention(args.hidden_dimension, args.hidden_dimension, heads=args.semantic_head, dropout=self.dropout)
        self.cross_img2node = CrossModalAttention(args.hidden_dimension, args.hidden_dimension, heads=args.semantic_head, dropout=self.dropout)
        # Gated fusion
        self.fusion = GatedFusion(args.hidden_dimension, args.hidden_dimension)
        # Relational graph layers
        self.RGT1 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=self.dropout)
        self.RGT2 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=self.dropout)
        # Classification head
        self.head = nn.Sequential( nn.Linear(args.hidden_dimension, args.out_channel), nn.LeakyReLU(), nn.Dropout(self.dropout), nn.Linear(args.out_channel, args.out_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features, edge_index, edge_type, img=None):
        txt_feats = self.text_proj(features)
        img_feats = self.img_proj(img) if img is not None else torch.zeros_like(txt_feats)
        # Cross-modal attention
        node_enh = self.cross_node2img(txt_feats, img_feats)
        img_enh  = self.cross_img2node(img_feats, txt_feats)
        # Gated fusion
        h, gate_vals = self.fusion(node_enh, img_enh)
        # Graph layers
        h = self.RGT1(h, edge_index, edge_type)
        h = torch.relu(h)
        h = self.RGT2(h, edge_index, edge_type)
        h = torch.relu(h)
        # Classification head
        out = self.head(h)
        return out
    
class RGT_multimodal_feedforward(nn.Module):
    def __init__(self, args):
        super(RGT_multimodal_feedforward, self).__init__()
        self.dropout = args.dropout
        # Text branch projection
        self.text_proj = nn.Sequential(nn.Linear(args.features_num, args.hidden_dimension), nn.LeakyReLU(), nn.Dropout(self.dropout))
        # Image branch projection
        self.img_proj = nn.Sequential(nn.Linear(args.img_feat_dim, args.hidden_dimension), nn.LeakyReLU(), nn.Dropout(self.dropout))
        # Relational graph layers
        self.RGT1 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=self.dropout)
        self.RGT2 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=self.dropout)
        # Classification head
        self.head = nn.Sequential( nn.Linear(args.hidden_dimension, args.out_channel), nn.LeakyReLU(), nn.Dropout(self.dropout), nn.Linear(args.out_channel, args.out_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features, edge_index, edge_type, img=None):
        txt_feats = self.text_proj(features)
        img_feats = self.img_proj(img) if img is not None else torch.zeros_like(txt_feats)
        h = txt_feats + img_feats
        # Graph layers
        h = self.RGT1(h, edge_index, edge_type)
        h = torch.relu(h)
        h = self.RGT2(h, edge_index, edge_type)
        h = torch.relu(h)
        # Classification head
        out = self.head(h)
        return out
