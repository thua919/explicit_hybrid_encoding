import tinycudann as tcnn
import torch
import torch.nn as nn
import numpy as np

class Decoder(nn.Module):
    def __init__(self,
                 bound=None,
                 voxel_size=None,
                 L=None,
                 F_entry=None,
                 log2_T=None,
                 b=None,
                 **kwargs):
        super().__init__()

        self.bound = torch.FloatTensor(bound)
        self.bound_dis = self.bound[:, 1] - self.bound[:, 0]
        self.max_dis = torch.ceil(torch.max(self.bound_dis))
        self.num_layers = 2
        self.pos_ch = 3
        self.hidden_dim = 32
        N_min = int(self.max_dis / voxel_size)

        # self.config['NeRFs']['space_resolution']['geo_block_size_fine'] = 0.02
        # self.config['NeRFs']['space_resolution']['geo_block_size_coarse'] = 0.24
        
        resolution_fine = list(map(int, (self.bound_dis / 0.02).tolist())) #self.config['NeRFs']['space_resolution']['geo_block_size_fine']).tolist()))
        resolution_coarse = list(map(int, (self.bound_dis / 0.24).tolist()))#self.config['NeRFs']['space_resolution']['geo_block_size_coarse']).tolist()))
        
        #per_level_scale = np.exp2(np.log2( max(resolution_fine) / max(resolution_coarse)) / (res_level - 1))
        self.embed = tcnn.Encoding(
            n_input_dims = 3,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": 1,
                    "n_features_per_level": 2,
                    "base_resolution": max(resolution_fine), #max(resolution_coarse),
                    "per_level_scale": 1,
                    "interpolation": "Linear"},
                dtype=torch.float
        )
        self.input_ch = self.embed.n_output_dims + self.pos_ch
        
        self.sdf_net = self.get_sdf_decoder()
        #self.sdf_net = sdf_net.to(device)
        self.hash_color_out = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=3,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F_entry,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,  # 1/base_resolution is the grid_size
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )
        
    def get_sdf_decoder(self):
        sdf_net = []
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.hidden_dim 
            if l == self.num_layers - 1:
                #if self.coupling:
                #    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                #else:
                out_dim = 1
            else:
                out_dim = self.hidden_dim 
            
            sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers - 1:
                sdf_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(sdf_net))

        
    def get_color(self, xyz):
        # xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.max_dis.to(xyz.device)
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        rgb = self.hash_color_out(xyz)
        return rgb

    def get_sdf(self, xyz):
        # xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.max_dis.to(xyz.device)
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        embed = self.embed(xyz)
        sdf = self.sdf_net(torch.cat([embed, xyz], dim=-1))
        return sdf

    def get_values(self, xyz):
        # xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.max_dis.to(xyz.device)
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        embed = self.embed(xyz)
        sdf = self.sdf_net(torch.cat([embed, xyz], dim=-1))
        rgb = self.hash_color_out(xyz)
        # outputs = torch.cat([rgb, sdf], dim=-1)
        return sdf, rgb

    def forward(self, xyz):
        sdf, rgb = self.get_values(xyz)
        return {
            'color': rgb[:, :3],
            'sdf': sdf[:, 0]
        }


if __name__ == "__main__":
    network = Decoder(1, 128, 16, skips=[], embedder='none', multires=0)
    print(network)
