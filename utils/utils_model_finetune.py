# utils_model_finetune.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists
from tqdm import tqdm
import time

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
    def forward(self, *input):
        x = self.first(*input)
        x = self.second(x)
        return x

class Network(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        radial_layers=1,
        radial_neurons=100,
        num_neighbors=1.0,
        num_nodes=1.0,
        reduce_output=True
    ):
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else o3.Irreps("0e")
        self.irreps_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        irreps = self.irreps_in
        act = {1: F.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        self.layers = nn.ModuleList()
        for _ in range(layers):
            irreps_scalars = o3.Irreps([
                (mul, ir) for mul, ir in self.irreps_hidden
                if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
            ])
            irreps_gated = o3.Irreps([
                (mul, ir) for mul, ir in self.irreps_hidden
                if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
            ])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated
            )
            conv = Convolution(
                irreps, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in,
                number_of_basis, radial_layers, radial_neurons, self.num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(
            Convolution(
                irreps, self.irreps_node_attr, self.irreps_edge_attr, self.irreps_out,
                number_of_basis, radial_layers, radial_neurons, self.num_neighbors
            )
        )

    def forward(self, data):
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_vec = data.edge_vec
        edge_length = edge_vec.norm(dim=1)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='gaussian',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        x = data.x
        z = data.z

        for layer in self.layers:
            x = layer(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.reduce_output:
            batch = data.batch if 'batch' in data else torch.zeros(data.pos.size(0), dtype=torch.long, device=x.device)
            x = scatter_mean(x, batch, dim=0)

        return x

class PeriodicNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        em_dim,
        device,
        feature_dim,
        num_properties,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        num_neighbors,
        reduce_output=True,
        out_dim=64
    ):
        super().__init__()
        self.device = device

        self.network = Network(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            irreps_node_attr=irreps_node_attr,
            layers=layers,
            mul=mul,
            lmax=lmax,
            max_radius=max_radius,
            num_neighbors=num_neighbors,
            reduce_output=reduce_output
        )
        self.em = nn.Linear(in_dim, em_dim)
        self.em_z = nn.Linear(em_dim, em_dim)

        self.batch_norm = nn.BatchNorm1d(feature_dim)
        self.fea_em = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU()
        )
        self.fea_update = nn.Sequential(
            nn.Linear(out_dim + feature_dim, out_dim + feature_dim),
            nn.SiLU(),
            nn.Linear(out_dim + feature_dim, out_dim + feature_dim),
            nn.SiLU(),
            nn.Linear(out_dim + feature_dim, num_properties)
        )

    def forward(self, data):
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em_z(data.x))
        output = self.network(data)

        feature = data.feature
        feature = self.fea_em(self.batch_norm(feature))
        output = torch.cat((output, feature), dim=1)
        output = self.fea_update(output)
        return output

def train(
    model,
    optimizer,
    dataloader_train,
    dataloader_valid,
    loss_fn,
    loss_fn_mae,
    run_name,
    max_iter=200,
    scheduler=None,
    device="cpu",
    patience=10
):
    model.to(device)
    history = {
        'train_loss': [],
        'train_mae': [],
        'valid_loss': [],
        'valid_mae': []
    }
    start_time = time.time()

    best_valid_loss = float('inf')
    no_improvement_steps = 0

    for step in range(1, max_iter + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for data in tqdm(dataloader_train, desc=f"Training Iteration {step}/{max_iter}"):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            data_y = data.y.view(data.num_graphs, -1)

            loss = loss_fn(output, data_y)
            loss_mae_val = loss_fn_mae(output, data_y)
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            train_loss += loss.item() * data.num_graphs
            train_mae += loss_mae_val.item() * data.num_graphs

        avg_train_loss = train_loss / len(dataloader_train.dataset)
        avg_train_mae = train_mae / len(dataloader_train.dataset)

        model.eval()
        valid_loss = 0.0
        valid_mae = 0.0
        with torch.no_grad():
            for data in tqdm(dataloader_valid, desc=f"Validation Iteration {step}/{max_iter}"):
                data = data.to(device)
                output = model(data)
                data_y = data.y.view(data.num_graphs, -1)

                loss = loss_fn(output, data_y)
                loss_mae_val = loss_fn_mae(output, data_y)
                valid_loss += loss.item() * data.num_graphs
                valid_mae += loss_mae_val.item() * data.num_graphs

        avg_valid_loss = valid_loss / len(dataloader_valid.dataset)
        avg_valid_mae = valid_mae / len(dataloader_valid.dataset)

        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(avg_train_mae)
        history['valid_loss'].append(avg_valid_loss)
        history['valid_mae'].append(avg_valid_mae)

        elapsed_time = time.time() - start_time
        print(f"Iteration {step}/{max_iter} | "
              f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f} | "
              f"Valid Loss: {avg_valid_loss:.4f}, Valid MAE: {avg_valid_mae:.4f} | "
              f"Elapsed Time: {elapsed_time:.2f}s")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            no_improvement_steps = 0
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history
            }, f"{run_name}_best.pth")
        else:
            no_improvement_steps += 1
            if no_improvement_steps >= patience:
                print(f"No improvement for {patience} steps, early stopping.")
                break

        if step % 10 == 0 or step == max_iter:
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history
            }, f"{run_name}_checkpoint_step_{step}.pth")

        if scheduler is not None:
            scheduler.step()

    torch.save({'model_state': model.state_dict(), 'history': history}, f"{run_name}_final.pth")
    print("Finetuning complete.")
    return history
