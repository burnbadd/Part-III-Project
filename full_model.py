from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean
from utils_new import *
import torch as torch
import torch as t
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import add_self_loops

class Ising_GCN(MessagePassing):
    def __init__(
        self, in_channels, out_channels, use_W=True, use_bias=False, device="cuda"
    ):
        super().__init__(aggr="add")  # "Add" aggregation (Step 5).

        self.use_W = use_W
        self.use_bias = use_bias

        if self.use_W:
            self.lin = Linear(in_channels, out_channels, bias=True)
        else:
            assert in_channels == out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_bias:
            self.bias = Parameter(t.Tensor(out_channels))

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_W:
            self.lin.reset_parameters()

        if self.use_bias:
            self.bias.data.zero_()

    def forward(self, x: Tensor, Jt_sp: Tensor):
        # x has shape [N, in_channels]
        # J_sp is the sparse Ising matrix with shape (N,N)
        # h_sp is the sparse Ising matrix with shape (N,1)

        N = Jt_sp.shape[0]

        if x.shape != (N, self.in_channels):
            raise ValueError(
                f"x should have dimensions {(N,self.in_channels)} but recieved dimensions {x.shape} instead"
            )

        # calculating the abs mean of each row, to add self loop
        Jt_sp_ind, Jt_sp_val = iv_from_sp(Jt_sp)

        row, col = Jt_sp_ind
        abs_val = Jt_sp_val.abs()

        # the abs mean of the entries in each row
        abs_mean = scatter_mean(src=abs_val, index=row,
                                dim_size=Jt_sp.size()[0])

        # Jt_sp with self-loop
        Jtsl_i, Jtsl_v = add_custom_diag(
            Jt_sp_ind, Jt_sp_val, custom_diag=abs_mean, size=N)

        # symmetrical normalise the value for the propagation matrix (norm_ij)
        prop_values = sym_norm_abs(Jtsl_i, Jtsl_v, size=N)

        # Linearly transform node feature matrix.
        if self.use_W:
            x = self.lin(x)

        # Start propagating messages
        out = self.propagate(edge_index=Jtsl_i, x=x, prop_val=prop_values)

        # Apply a final bias vector.
        if self.use_bias:
            out += self.bias

        return out

    def message(self, x_j, prop_val):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return prop_val.view(-1, 1) * x_j


class feature_generation(nn.Module):
    def __init__(self, feature_dim, device="cuda", drop_p=0.1) -> None:
        super().__init__()

        self.device = device

        # pure convolution, no linear transform
        self.conv1 = Ising_GCN(1, 1,
                               use_W=False, use_bias=False, device=device)
        self.linear = nn.Linear(1, feature_dim, bias=True)
        # normal GCN
        self.conv2 = Ising_GCN(feature_dim, feature_dim*2,
                               use_bias=True, device=device)
        self.conv3 = Ising_GCN(feature_dim*2, feature_dim,
                               use_bias=True, device=device)

        self.non_linear = nn.LeakyReLU(0.1)

        self.dropout = nn.Dropout(p=drop_p)
        self.norm1 = nn.BatchNorm1d(num_features=feature_dim)
        self.norm2 = nn.BatchNorm1d(num_features=feature_dim)
        self.norm3 = nn.BatchNorm1d(num_features=feature_dim)

    def forward(self, Jt_sp: Tensor):
        N = Jt_sp.shape[0]

        # x_init is (N, feature_dim)
        # x_init = self.initialisation(t.ones(N, 1, device=self.device))
        x_init = t.ones(N,1, device=self.device)
        x = self.conv1(x_init, Jt_sp)
        x = self.linear(x)
        # x = self.norm1(x)
        x = self.non_linear(x)
        x = self.dropout(x)

        x = self.conv2(x, Jt_sp)
        # x = self.norm2(x)
        x = self.non_linear(x)
        x = self.dropout(x)

        x = self.conv3(x, Jt_sp)
        x = self.non_linear(x)

        return x


class attention_prediction_asym(nn.Module):
    def __init__(self, feature_dim, device="cuda", drop_p=0.1, collapse_bias=1) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.layer1_input_dim = feature_dim*2
        self.layer1_output_dim = int((feature_dim*2)**0.5)
        self.layer2_input_dim = self.layer1_output_dim + 1
        self.layer2_output_dim = self.layer2_input_dim

        self.layer1 = nn.Linear(in_features=self.layer1_input_dim, out_features=self.layer1_output_dim)
        self.layer2 = nn.Linear(in_features=self.layer2_input_dim, out_features=self.layer2_output_dim)
        self.collapse = nn.Linear(in_features=self.layer2_output_dim, out_features=1)

        if collapse_bias!= None:
            self.collapse.bias.data.fill_(collapse_bias)

        self.nonlinear = nn.LeakyReLU(0.1)
        self.device = device
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, Jt_ind, Jt_val, f, N):
        i, j = Jt_ind
        deg_abs = degree(Jt_ind, Jt_val.abs(), N)
        Jt_val_abs_norm = Jt_val.abs()/t.index_select(deg_abs, dim=0, index=i)

        # [f_i || f_j]
        f_i = t.index_select(f, dim=0, index=i)
        f_j = t.index_select(f, dim=0, index=j)
        input = t.cat([f_i, f_j], dim=1)

        #layer 1
        x = self.layer1(input)
        x = self.nonlinear(x)

        #layer 2
        x = t.cat([x, Jt_val_abs_norm.unsqueeze(1)], dim=1)
        x = self.layer2(x)
        x = self.nonlinear(x)

        e_values = self.collapse(x).squeeze(1)

        a_values = norm_attention_from_raw(e_values, Jt_ind)

        return a_values


class diffusion_const_field(nn.Module):
    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.device = device

    def forward(self, adj_ind: Tensor, pol_a_val: Tensor, N_1: int, dt=0.1, T=10):

        # diffusion operator = I + dt* del_t_operator = (1-dt)*I + dt*(Pol_atten)
        diff_ind, diff_val = add_self_loops(
            edge_index=adj_ind, edge_attr=(dt * pol_a_val), fill_value=(1 - dt)
        )

        # indices
        diff_i, diff_j = diff_ind

        # clear the first row
        diff_val[diff_i == 0] = 0  # type: ignore

        # make (0,0) in the first row a 1 to keep the external spin
        diff_val[(diff_i == 0) * (diff_j == 0)] = 1  # type: ignore

        # diffusion linear operator
        diff_sp = sp_from_iv(diff_ind, diff_val, N=N_1)

        total_steps = int(T / dt)

        spins = t.ones((N_1, 1)).to(self.device)

        for _ in range(total_steps):
            spins = t.sparse.mm(diff_sp, spins)

            if total_steps%(int(5/dt)) == 0:
                # only the sites, excluding the field spin at 0
                site_spins = spins[1:]
                # normalising
                normed_site_spins = site_spins/(site_spins.abs().max())
                # adding back the field spin
                normed_all_spins = t.cat([t.ones((1, 1)), normed_site_spins], dim=0)


        # only the sites, excluding the field spin at 0
        site_spins = spins[1:]
        # normalising
        normed_site_spins = site_spins/(site_spins.abs().max())
        # adding back the field spin
        normed_all_spins = t.cat([t.ones((1, 1)), normed_site_spins], dim=0)

        return normed_all_spins


class full_model(nn.Module):
    def __init__(self, feature_dim, f_drop, a_drop, atten_bias, device="cuda") -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.m_fgen = feature_generation(
            feature_dim=feature_dim, device=device, drop_p=f_drop
        )

        self.m_atten = attention_prediction_asym(
            feature_dim=feature_dim, device=device, drop_p=a_drop, collapse_bias=atten_bias
        )

        self.m_diff = diffusion_const_field(device=device)

    def forward(self, Jt_sp: Tensor, T):
        # N+1
        N_1 = Jt_sp.shape[0]

        adj_ind, Jt_val = iv_from_sp(Jt_sp)

        # feature generated
        f = self.m_fgen(Jt_sp)

        # softmax normalised a_ij
        a_val = self.m_atten(adj_ind,Jt_val, f, N_1)

        # polarised attention, polarisation = opposite of Jt
        pol_a_val = -t.sign(Jt_val) * a_val

        diff_result = self.m_diff(adj_ind, pol_a_val, N_1=N_1, T=T)

        return diff_result