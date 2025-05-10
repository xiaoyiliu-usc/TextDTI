import numpy as np #type:ignore
from math import sqrt
import torch #type:ignore
import torch.nn as nn #type:ignore
from torch.nn import Parameter #type:ignore
import torch.nn.functional as F #type:ignore
import torch.optim as optim #type:ignore
import pickle
import timeit
from sklearn.metrics import roc_auc_score, precision_score, precision_recall_curve, auc, accuracy_score, f1_score, matthews_corrcoef #type:ignore

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.05):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.05):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):

        B, H, L, E = K.shape
        _, _, S, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
        indx_sample = torch.randint(L, (S, sample_k))
        K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), indx_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V)
        return context_in

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        queries = queries.view(B, H, L, -1)
        keys = keys.view(B, H, S, -1)
        values = values.view(B, H, S, -1)

        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()

        scores_top, index = self._prob_QK(queries, keys, u, U)
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L)
        context = self._update_context(context, values, scores_top, index, L, attn_mask)

        return context.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, -1)

        return self.out_projection(out)

class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=o_channel, out_channels=growth_rate, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(o_channel)
        self.drop_rate = 0.05

    def forward(self, x):
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)

        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in
             range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1,
                                  padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05):
        super(Decoder, self).__init__()
        self.atten0 = AttentionLayer(ProbAttention(None, 3, dropout), d_model,
                                     n_heads)
        self.atten1 = AttentionLayer(ProbAttention(None, 3, dropout), d_model,
                                     n_heads)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, xs, xd, xp):
        xp = self.drop(self.atten0(xp, xd, xd, None))
        xd = self.drop(self.atten1(xd, xp, xp, None))
        xs = xs + torch.cat((xd, xp), dim=1)
        return xs
        

class Fusion(nn.Module):
    def __init__(self, hidden1, hidden2, dropout=0.05):
        super(Fusion, self).__init__()
        self.si_L = nn.Sigmoid()
        self.si_S = nn.Sigmoid()
        self.so_f = nn.Sigmoid()
        self.combine = nn.Linear(128 * 4, 128)
        self.ln = nn.LayerNorm(128)
        self.drop = nn.Dropout(p=dropout)


    def forward(self, LM_fea, Sty_fea):

        Sty_fea_norm = Sty_fea * (abs(torch.mean(LM_fea))/abs(torch.mean(Sty_fea)))
        f_h = torch.cat((LM_fea.unsqueeze(1), Sty_fea_norm.unsqueeze(1)), dim=1)

        f_att = torch.mean(f_h, dim=1)
        f_att = self.so_f(f_att)
        fus_fea = torch.cat((LM_fea, Sty_fea, LM_fea * f_att, Sty_fea * f_att), dim=1)
        fus_fea = self.combine(fus_fea)

        return fus_fea


class DT_LeNet(nn.Module):
    def __init__(self, hidden, dropout, classes, layers):
        super(DT_LeNet, self).__init__()
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(hidden)

        self.FC_combs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.FC_down = nn.Linear(hidden, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_feature):
        dti_feature = dti_feature.permute(0, 2, 1)
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        GRL_feature = dti_feature.clone()
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti, dti_feature, GRL_feature


class GRL(nn.Module):

    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - 1)
        return -coeff * gradOutput


class Discriminator(nn.Module):
    def __init__(self, max_iter, source_number, device):
        super(Discriminator, self).__init__()
        self.fc1 = Parameter(torch.Tensor(256, 128))
        self.fc2 = Parameter(torch.Tensor(75, 128))
        self.fc3 = Parameter(torch.Tensor(128, source_number))
        self.drop_lm = nn.Dropout(0.0)
        self.drop_sty = nn.Dropout(0.0)
        self.drop = nn.Dropout(0.5)
        self.grl_layer = GRL(10000)
        self.grl_layer2 = GRL(100)
        self.source_number = source_number
        self.device = device

    def forward(self, feature1, feature2):
        if self.source_number > 2:
            adversarial_out1 = self.grl_layer(self.drop_lm(feature1.detach()))
            adversarial_out2 = self.grl_layer2(self.drop_sty(feature2))
            adversarial_out1 = torch.matmul(adversarial_out1,  nn.init.xavier_uniform_(self.fc1))
            adversarial_out2 = torch.matmul(adversarial_out2, nn.init.xavier_uniform_(self.fc2))
            adversarial_out = adversarial_out2 * torch.sigmoid(adversarial_out1)
            adversarial_out = torch.matmul(self.drop(torch.relu(adversarial_out)), nn.init.xavier_uniform_(self.fc3))
        else:
            adversarial_out = torch.zeros((feature1.shape[0], 1), device=self.device)
        return adversarial_out


class Contrast_Fusion(nn.Module):
    def __init__(self, dropout=0.05):
        super(Contrast_Fusion, self).__init__()
        self.so_L = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, LM_fea, Sty_fea):
        LM_att = self.so_L(LM_fea)
        fus_fea = LM_fea + Sty_fea * LM_att
        return fus_fea

class ContrastLoss(nn.Module):
    def __init__(self, source_number):
        super(ContrastLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.source_number = source_number
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        if self.source_number > 2:
            contrast_label = contrast_label.float()
            anchor_fea = anchor_fea.detach()
            loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
            loss = loss * contrast_label
        else:
            loss = 0.0 * contrast_label
        return loss.mean()

class Smooth_loss(nn.Module):

    def __init__(self, smoothing=0.05):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class TEXT_DTI(nn.Module):
    def __init__(self, layer_gnn, device, source_number, hidden1=256, hidden2=75, n_layers=1, attn_heads=1,
                 dropout=0.0):
        super(TEXT_DTI, self).__init__()
        self.embed_protein = nn.Embedding(26, hidden2)
        self.W_dnn = nn.ModuleList([nn.Linear(hidden2, hidden2)
                                    for _ in range(layer_gnn)])
        self.W_pnn = nn.ModuleList([nn.Linear(hidden2, hidden2)
                                    for _ in range(layer_gnn)])

        self.gnn_act = nn.GELU()
        self.G_A = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden2, out_channels=hidden2, kernel_size=3, padding=1, groups=hidden2, bias=False)
             for _ in range(layer_gnn)])
        self.encoder_protein_GNN = Encoder(hidden2, hidden2, 15, 5, groups=1, pad1=15, pad2=7)
        self.bn_A = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_B = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_C = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_D = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.gnn_drop = nn.Dropout(p=0.05)
        self.gnn_output = nn.Linear(hidden2, hidden2, bias=False)

        self.encoder_protein_LM = Encoder(768, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.encoder_drug = Encoder(768, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.Informer_blocks = nn.ModuleList(
            [AttentionLayer(ProbAttention(None, 5, 0), hidden1, attn_heads),
             AttentionLayer(ProbAttention(None, 5, 0), hidden1, attn_heads)])

        self.soft_1 = nn.Softmax(-1)
        self.soft_2 = nn.Softmax(-1)
        self.soft_3 = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.layer_gnn = layer_gnn
        self.hidden = hidden1
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.fusion = Fusion(hidden1, hidden2)
        self.FC_out1 = DT_LeNet(hidden1, 0.1, 2, 3)
        self.FC_out2 = DT_LeNet(hidden2, 0.0, 2, 3)
        self.DTI_feature = nn.ModuleList([nn.Linear(128, 128) for _ in range(2)])
        self.act = nn.ReLU()
        self.DTI_Pre = nn.Linear(128, 2)
        self.dis = Discriminator(100, source_number, device)
        self.Contrast_Fusion = Contrast_Fusion()
        self.source_number = source_number
        self.lamda = Parameter(torch.Tensor([0.8, 0.1, 0.1]))


    def Style_Exract(self, df, da, pf, layer):
        for i in range(layer):
            ds = self.gnn_act(self.W_dnn[i](df))
            ps = self.gnn_act(self.W_pnn[i](pf))
            dg_A = self.bn_A[i](self.G_A[i](ps.permute(0, 2, 1))).permute(0, 2, 1)
            G_CB = torch.matmul(pf, df.permute(0, 2, 1))
            dg_B = self.bn_B[i](torch.matmul(self.soft_1(G_CB), ds).permute(0, 2, 1)).permute(0, 2, 1)
            dg_C = self.bn_C[i](torch.matmul(da, ds).permute(0, 2, 1)).permute(0, 2, 1)
            G_BC = torch.matmul(df, pf.permute(0, 2, 1))
            dg_D = self.bn_D[i](torch.matmul(self.soft_2(G_BC), ps).permute(0, 2, 1)).permute(0, 2, 1)
            pf = dg_A + dg_B + pf
            df = dg_C + dg_D + df
        dt = torch.cat((pf, df), dim=1)
        return dt

    def forward(self, inputs, rand_idx):

        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        proteins_acids_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)
        molecule_smiles_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)

        DT_1D_Feature = torch.cat((proteins_acids_LM, molecule_smiles_LM), 1)
        DT_1D_P_att = self.dropout(self.Informer_blocks[0](DT_1D_Feature, proteins_acids_LM, proteins_acids_LM, None))
        DT_1D_D_att = self.dropout(self.Informer_blocks[1](DT_1D_Feature, molecule_smiles_LM, molecule_smiles_LM, None))
        DT_1D_Feature = DT_1D_P_att + DT_1D_D_att

        proteins_acids_GNN = torch.zeros((proteins.shape[0], proteins.shape[1], 75), device=self.device)
        DT_2D_Feature = torch.zeros((N, 1300, 75), device=self.device)
        for i in range(N):
            proteins_acids_GNN[i, :, :] = self.embed_protein(torch.LongTensor(proteins[i].to('cpu').numpy()).cuda())
        proteins_acids_GNN = self.encoder_protein_GNN(proteins_acids_GNN.permute(0, 2, 1)).permute(0, 2, 1)
        DT_2D_F = self.Style_Exract(molecule_atoms, molecule_adjs, proteins_acids_GNN, self.layer_gnn)
        t = DT_2D_F.shape[1]
        if t < 1300:
            DT_2D_Feature[:, 0:t, :] = DT_2D_F
        else:
            DT_2D_Feature = DT_2D_F[:, 0:1300, :]

        dti1d, dti1d_feature, LM_feature = self.FC_out1(DT_1D_Feature)
        dti2d, dti2d_feature, GRL_feature = self.FC_out2(DT_2D_Feature)
        dis_invariant = self.dis(LM_feature, GRL_feature)
        DTI = self.fusion(dti1d_feature, dti2d_feature)
        DTI_normal = DTI.clone()
        DTI_shuffle = self.fusion(dti1d_feature, dti2d_feature[rand_idx])
        for i in range(2):
            DTI = self.act(self.DTI_feature[i](DTI))
        DTI = self.DTI_Pre(DTI)
        lam_DTI = self.lamda[0] * DTI.detach() + self.lamda[1] * dti1d.detach() + self.lamda[2] * dti2d.detach()
        return DTI, dti1d, dti2d, dti1d_feature, dti2d_feature, DTI_normal, DTI_shuffle, lam_DTI, self.lamda, dis_invariant

    def predict(self, res):
        if res > 0.5:
            result = 1
        else:
            result = 0
        return result

    def __call__(self, data, epoch=1, train=True):
        l1 = 1
        l2 = 1
        l3 = 1.25
        l4 = 1
        l5 = 1

        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        SID = torch.LongTensor(SID.to('cpu').numpy()).cuda()
        LACE = Smooth_loss()
        rand_idx = torch.randperm(correct_interaction.shape[0])
        contrast_label = correct_interaction.long() == correct_interaction[rand_idx].long()
        contrast_label = torch.where(contrast_label == True, 1, -1)
        contrast_loss= ContrastLoss(self.source_number)
        protein_drug_interaction, dti1d, dti2d, dti1d_feature, dti2d_feature, DTI_normal, DTI_shuffle, lam_DTI, lamda, dis_invariant = self.forward(inputs, rand_idx)  # , dis_invariant
        if train:
            loss1 = F.cross_entropy(protein_drug_interaction, correct_interaction)
            loss2 = F.cross_entropy(dti1d, correct_interaction)
            loss3 = F.cross_entropy(dti2d, correct_interaction)
            loss5 = LACE(dis_invariant, SID)
            loss6 = contrast_loss(DTI_normal, DTI_shuffle, contrast_label)
            return loss1 * l1 + loss2 * l2 + loss3 * l3 + loss5 * l4 + loss6 * l5
        else:
            correct_labels = correct_interaction
            ys1 = F.softmax(protein_drug_interaction * 0.4 + dti1d * 0.27 + dti2d * 0.33, 1)
            ys2 = F.softmax(dti1d, 1)
            ys3 = F.softmax(dti2d, 1)
            return correct_labels, ys1, ys2, ys3, dti1d_feature, dti2d_feature, DTI_normal

def load_tensor(file_name, dtype, device):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def train_data_load(dataset, device, DTI=True):

    molecule_words_train = load_tensor('.//datasets//' + dataset + '//train//molecule_words', torch.LongTensor, device)
    molecule_atoms_train = load_tensor('.//datasets//' + dataset + '//train//molecule_atoms', torch.LongTensor, device)
    molecule_adjs_train = load_tensor('.//datasets//' + dataset + '//train//molecule_adjs', torch.LongTensor, device)
    proteins_train = load_tensor('.//datasets//' + dataset + '//train//proteins', torch.LongTensor, device)
    sequence_train = np.load('.//datasets//' + dataset + '//train//sequences.npy')
    smiles_train = np.load('.//datasets//' + dataset + '//train//smiles.npy')
    if DTI == True:
        interactions_train = load_tensor('.//datasets//' + dataset + '//train//interactions', torch.LongTensor, device)
    else:
        interactions_train = load_tensor('.//datasets//' + dataset + '//train//affinity', torch.FloatTensor, device)

    with open('.//datasets//' + dataset + '//train//p_LM.pkl', 'rb') as p:
        p_LM = pickle.load(p)

    with open('.//datasets//' + dataset + '//train//d_LM.pkl', 'rb') as d:
        d_LM = pickle.load(d)

    return molecule_words_train, molecule_atoms_train, molecule_adjs_train, proteins_train, sequence_train, smiles_train, p_LM, d_LM, interactions_train

def test_data_load(dataset, device, DTI=True):
    molecule_words_test = load_tensor('.//datasets//' + dataset + '//test//molecule_words', torch.LongTensor, device)
    molecule_atoms_test = load_tensor('.//datasets//' + dataset + '//test//molecule_atoms', torch.LongTensor, device)
    molecule_adjs_test = load_tensor('.//datasets//' + dataset + '//test//molecule_adjs', torch.LongTensor, device)
    proteins_test = load_tensor('.//datasets//' + dataset + '//test//proteins', torch.LongTensor, device)
    sequence_test = np.load('.//datasets//' + dataset + '//test//sequences.npy')
    smiles_test = np.load('.//datasets//' + dataset + '//test//smiles.npy')
    if DTI == True:
        interactions_test = load_tensor('.//datasets//' + dataset + '//test//interactions', torch.LongTensor, device)
    else:
        interactions_test = load_tensor('.//datasets//' + dataset + '//test//affinity', torch.FloatTensor, device)


    with open('.//datasets//' + dataset + '//test//p_LM.pkl', 'rb') as p:
        p_LM = pickle.load(p)

    with open('.//datasets//' + dataset + '//test//d_LM.pkl', 'rb') as d:
        d_LM = pickle.load(d)
    return molecule_words_test, molecule_atoms_test, molecule_adjs_test, proteins_test, sequence_test, smiles_test, p_LM, d_LM, interactions_test

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def Source_ID(ID, length):
    source_label = torch.ones(length) * ID
    return source_label

def merge(train_list, test_list, device, DTI=True):
    molecule_words_trains, molecule_atoms_trains, molecule_adjs_trains, proteins_trains, sequence_trains, smiles_trains, interactions_trains = [], [], [], [], [], [], []
    molecule_words_tests, molecule_atoms_tests, molecule_adjs_tests, proteins_tests, sequence_tests, smiles_tests, interactions_tests = [], [], [], [], [], [], []
    p_LMs, d_LMs = {}, {}
    train_source = []
    ID = 0
    for dataset in train_list:
        molecule_words_train, molecule_atoms_train, molecule_adjs_train, proteins_train, sequence_train, smiles_train, p_LM, d_LM, interactions_train = train_data_load(dataset, device, DTI)
        molecule_words_trains.extend(molecule_words_train)
        molecule_atoms_trains.extend(molecule_atoms_train)
        molecule_adjs_trains.extend(molecule_adjs_train)
        proteins_trains.extend(proteins_train)
        sequence_trains.extend(sequence_train)
        smiles_trains.extend(smiles_train)
        p_LMs.update(p_LM)
        d_LMs.update(d_LM)
        interactions_trains.extend(interactions_train)
        length = len(molecule_words_train)
        train_source.extend(Source_ID(ID, length))
        ID += 1

    for dataset in test_list:
        molecule_words_test, molecule_atoms_test, molecule_adjs_test, proteins_test, sequence_test, smiles_test, p_LM, d_LM, interactions_test = test_data_load(dataset, device, DTI)
        molecule_words_tests.extend(molecule_words_test)
        molecule_atoms_tests.extend(molecule_atoms_test)
        molecule_adjs_tests.extend(molecule_adjs_test)
        proteins_tests.extend(proteins_test)
        sequence_tests.extend(sequence_test)
        smiles_tests.extend(smiles_test)
        p_LMs.update(p_LM)
        d_LMs.update(d_LM)
        interactions_tests.extend(interactions_test)

    train_dataset = list(zip(molecule_words_trains, molecule_atoms_trains, molecule_adjs_trains, proteins_trains, sequence_trains, smiles_trains, interactions_trains, train_source))
    train_dataset = shuffle_dataset(train_dataset, 1234)

    test_dataset = list(zip(molecule_words_tests, molecule_atoms_tests, molecule_adjs_tests, proteins_tests, sequence_tests, smiles_tests, interactions_tests))
    test_dataset = shuffle_dataset(test_dataset, 1234)

    return train_dataset, test_dataset, p_LMs, d_LMs

def data_load(data_select, device):

    if data_select == "B_to_B":
        train_list = ["BindingDB"]
        test_list = ["BindingDB"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "D_to_D":
        train_list = ["Drugbank"]
        test_list = ["Drugbank"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "H_to_H":
        train_list = ["human"]
        test_list = ["human"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "C_to_C":
        train_list = ["celegans"]
        test_list = ["celegans"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "B_D_H_to_C":
        train_list = ["BDH"]
        test_list = ["celegans"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "B_D_C_to_H":
        train_list = ["BDC"]
        test_list = ["human"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "B_H_C_to_D":
        train_list = ["BHC"]
        test_list = ["Drugbank"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "D_H_C_to_B":
        train_list = ["DHC"]
        test_list = ["BindingDB"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "Da_to_Da":
        train_list = ["Davis"]
        test_list = ["Davis"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "K_to_K":
        train_list = ["KIBA"]
        test_list = ["KIBA"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "Da_to_K":
        train_list = ["Davis"]
        test_list = ["KIBA"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "K_to_Da":
        train_list = ["KIBA"]
        test_list = ["Davis"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)
    else:
        train_list = ["human"]
        test_list = ["human"]
        train_dataset, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device)

    return train_dataset, test_dataset, p_LMs, d_LMs

torch.multiprocessing.set_start_method('spawn')

def pack(molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, p_LMs, d_LMs, device, sources=None):

    proteins_len = 1200
    words_len = 100
    atoms_len = 0
    p_l = 1200
    d_l = 100
    N = len(molecule_atoms)
    molecule_words_new = torch.zeros((N, words_len), device=device)
    i = 0
    for molecule_word in molecule_words:
        molecule_word_len = molecule_word.shape[0]
        if molecule_word_len <= 100:
            molecule_words_new[i, :molecule_word_len] = molecule_word
        else:
            molecule_words_new[i] = molecule_word[0:100]
        i += 1

    atom_num = []
    for atom in molecule_atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    molecule_atoms_new = torch.zeros((N, atoms_len, 75), device=device)
    i = 0
    for atom in molecule_atoms:
        a_len = atom.shape[0]
        molecule_atoms_new[i, :a_len, :] = atom
        i += 1

    molecule_adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in molecule_adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        molecule_adjs_new[i, :a_len, :a_len] = adj
        i += 1

    proteins_new = torch.zeros((N, proteins_len), device=device)
    i = 0
    for protein in proteins:
        if protein.shape[0] > 1200:
            protein = protein[0:1200]
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1

    protein_LMs = []
    molecule_LMs = []
    for sequence in sequences:
        protein_LMs.append(p_LMs[sequence])

    for smile in smiles:
        molecule_LMs.append(d_LMs[smile])

    protein_LM = torch.zeros((N, p_l, 768), device=device)
    molecule_LM = torch.zeros((N, d_l, 768), device=device)
    for i in range(N):
        C_L = molecule_LMs[i].shape[0]
        if C_L >= 100:
            molecule_LM[i, :, :] = torch.tensor(molecule_LMs[i][0:100, :]).to(device)
        else:
            molecule_LM[i, :C_L, :] = torch.tensor(molecule_LMs[i]).to(device)
        P_L = protein_LMs[i].shape[0]
        if P_L >= 1200:
            protein_LM[i, :, :] = torch.tensor(protein_LMs[i][0:1200, :]).to(device)
        else:
            protein_LM[i, :P_L, :] = torch.tensor(protein_LMs[i]).to(device)

    labels_new = torch.zeros(N, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    if sources != None:
        sources_new = torch.zeros(N, device=device)
        i = 0
        for source in sources:
            sources_new[i] = source
            i += 1
    else:
        sources_new = torch.zeros(N, device=device)

    return molecule_words_new, molecule_atoms_new, molecule_adjs_new, proteins_new, protein_LM, molecule_LM, labels_new, sources_new


class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size

    def train(self, dataset, p_LMs, d_LMs, epoch):
        np.random.shuffle(dataset)
        N = len(dataset)

        loss_total = 0
        i = 0
        self.optimizer.zero_grad()

        molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            molecule_word, molecule_atom, molecule_adj, protein, sequence, smile, label, source = data
            molecule_words.append(molecule_word)
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            proteins.append(protein)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)
            sources.append(source)

            if i % self.batch_size == 0 or i == N:
                if len(molecule_words) != 1:
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = pack(molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, p_LMs, d_LMs, device, sources)
                    data = (molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources)
                    loss = self.model(data, epoch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
                else:
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
            else:
                continue

            if i % self.batch_size == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()

        return loss_total

class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset, p_LMs, d_LMs):
        N = len(dataset)
        T, S, Y, S2, Y2, S3, Y3 = [], [], [], [], [], [], []
        i = 0
        molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels = [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            molecule_word, molecule_atom, molecule_adj, protein, sequence, smile, label = data
            molecule_words.append(molecule_word)
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            proteins.append(protein)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)

            if i % self.batch_size == 0 or i == N:
                molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, _ = pack(molecule_words, molecule_atoms,
                                                                                       molecule_adjs, proteins, sequences, smiles, labels, p_LMs, d_LMs,
                                                                                       device)
                data = (molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, _)
                correct_labels, ys1, ys2, ys3, _, _, _ = self.model(data, train=False)
                correct_labels = correct_labels.to('cpu').data.numpy()
                ys1 = ys1.to('cpu').data.numpy()
                ys2 = ys2.to('cpu').data.numpy()
                ys3 = ys3.to('cpu').data.numpy()
                predicted_labels1 = list(map(lambda x: np.argmax(x), ys1))
                predicted_scores1 = list(map(lambda x: x[1], ys1))
                predicted_labels2 = list(map(lambda x: np.argmax(x), ys2))
                predicted_scores2 = list(map(lambda x: x[1], ys2))
                predicted_labels3 = list(map(lambda x: np.argmax(x), ys3))
                predicted_scores3 = list(map(lambda x: x[1], ys3))

                for j in range(len(correct_labels)):
                    T.append(correct_labels[j])
                    Y.append(predicted_labels1[j])
                    S.append(predicted_scores1[j])
                    Y2.append(predicted_labels2[j])
                    S2.append(predicted_scores2[j])
                    Y3.append(predicted_labels3[j])
                    S3.append(predicted_scores3[j])

                molecule_words, molecule_atoms, molecule_adjs, proteins,  sequences, smiles, labels = [], [], [], [], [], [], []
            else:
                continue

        AUC = roc_auc_score(T, S)
        AUC2 = roc_auc_score(T, S2)
        AUC3 = roc_auc_score(T, S3)
        precision = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        F1 = f1_score(T, Y)
        MCC = matthews_corrcoef(T, Y)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return AUC, precision, PRC, AUC2, AUC3, ACC, F1, MCC

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

    def export_features(self, dataset, p_LMs, d_LMs, device, split_name='train'):
        self.model.eval()
        all_labels = []
        all_d1d   = []
        all_d2d   = []
        all_norm  = []

        i = 0
        batch_size = self.batch_size
        molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels = [], [], [], [], [], [], []

        for data in dataset:
            i += 1
            mw, ma, madj, p, seq, sm, label = data[:7]
            molecule_words.append(mw)
            molecule_atoms.append(ma)
            molecule_adjs.append(madj)
            proteins.append(p)
            sequences.append(seq)
            smiles.append(sm)
            labels.append(label)

            
            if i % batch_size == 0 or i == len(dataset):
                
                batch = pack(molecule_words, molecule_atoms, molecule_adjs,
                             proteins, sequences, smiles, labels,
                             p_LMs, d_LMs, device)
                
                with torch.no_grad():
                    out = self.model(batch, train=False)
                
                correct_labels = out[0].cpu().numpy()
                d1d_feat       = out[4].detach().cpu().numpy()
                d2d_feat       = out[5].detach().cpu().numpy()
                dti_norm       = out[6].detach().cpu().numpy()

                all_labels.append(correct_labels)
                all_d1d.append(d1d_feat)
                all_d2d.append(d2d_feat)
                all_norm.append(dti_norm)

                molecule_words, molecule_atoms, molecule_adjs = [], [], []
                proteins, sequences, smiles, labels = [], [], [], []

        all_labels = np.concatenate(all_labels, axis=0)
        all_d1d    = np.concatenate(all_d1d,    axis=0)
        all_d2d    = np.concatenate(all_d2d,    axis=0)
        all_norm   = np.concatenate(all_norm,   axis=0)

        np.save(f'{split_name}_labels.npy',       all_labels)
        np.save(f'{split_name}_dti1d_feature.npy',all_d1d)
        np.save(f'{split_name}_dti2d_feature.npy',all_d2d)
        np.save(f'{split_name}_DTI_normal.npy',   all_norm)

        print(f'[+] Saved {split_name} features: ', 
              all_labels.shape, all_d1d.shape, all_d2d.shape, all_norm.shape)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_select = "B_D_H_to_C"
    iteration = 10
    decay_interval = 5
    batch_size = 16
    lr = 5e-4
    weight_decay = 0.07
    lr_decay = 0.5
    layer_gnn = 3
    source_number = 1
    drop = 0.05
    setting = "B_D_H_to_C"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset_train, dataset_test, p_LMs, d_LMs = data_load(data_select, device)
    setup_seed(1000)
    model = TEXT_DTI(layer_gnn=layer_gnn, source_number=source_number, device=device, dropout=drop).to(device)
    trainer = Trainer(model, batch_size, lr, weight_decay)
    tester = Tester(model, batch_size)
    file_model = './/best_model//' + setting

    AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
            'AUC_test\tPrecision_test\tAUPR_test\tACC_test\tF1_test\tMCC_test\tAUC_LM\tAUC_Sty')

    print('Running...')
    print(AUCs)
    start = timeit.default_timer()
    auc1 = 0
    for epoch in range(1, iteration + 1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        loss_train = trainer.train(dataset_train, p_LMs, d_LMs, epoch)
        AUC_test, precision_test, recall_test, AUC2, AUC3, ACC_test, F1_test, MCC_test = tester.test(dataset_test, p_LMs, d_LMs)
        end = timeit.default_timer()
        time = end - start
        AUCs = [epoch, time, loss_train,
                AUC_test, precision_test, recall_test, ACC_test, F1_test, MCC_test, AUC2, AUC3]
        print('\t'.join(map(str, AUCs)))
        if auc1 < AUC_test:
            auc1 = AUC_test
            tester.save_model(model, file_model)

    print("Exporting all features…")
    tester.export_features(dataset_train, p_LMs, d_LMs, device, split_name='train')
    tester.export_features(dataset_test,  p_LMs, d_LMs, device, split_name='test')
