from typing import Final, Tuple, Union

import torch
from torch import Tensor, nn

from torch_geometric import EdgeIndex
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import spmm


class GraphConvMLP(MessagePassing):
    SUPPORTS_FUSED_EDGE_INDEX: Final[bool] = True

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        hidden_channels: int,
        out_channels: int,
        aggr: str = 'add',
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # MLP for messages: MLP([s_i | s_j | w_ij])
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels[0] * 2 + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Final update MLP: MLP([s_i | agg])
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels[1] + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.message_mlp, self.update_mlp]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        self._x_target = x[1]  # s_i for message construction
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        combined = torch.cat([x[1], out], dim=-1)  # s_i | agg_msg
        return self.update_mlp(combined)

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            edge_weight = torch.ones(x_j.size(0), device=x_j.device)

        edge_weight = edge_weight.view(-1, 1)  # ensure shape (E, 1)
        m_ij_input = torch.cat([x_i, x_j, edge_weight], dim=-1)  # s_i | s_j | w_ij
        return self.message_mlp(m_ij_input)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'
