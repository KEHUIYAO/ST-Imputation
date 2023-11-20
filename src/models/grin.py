from tsl.nn.models.imputation import GRINModel
from torch import Tensor, LongTensor
from torch_geometric.typing import OptTensor
from tsl.utils.parser_utils import str_to_bool


class GrinModel(GRINModel):

    def forward(self, x: Tensor, edge_index: LongTensor,
                edge_weight: OptTensor = None, mask: OptTensor = None,
                u: OptTensor = None, **kwargs):
        return super().forward(x, edge_index, edge_weight, mask, u)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--hidden-size', type=int)
        parser.add_argument('--ff-size', type=int)
        parser.add_argument('--embedding-size', type=int, default=None)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--n-nodes', type=int, default=None)
        parser.add_argument('--kernel-size', type=int, default=2)
        parser.add_argument('--decoder-order', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.)
        parser.add_argument('--ff-dropout', type=float, default=0.)
        parser.add_argument('--merge-mode', type=str, default='mlp',
                            choices=['mlp', 'mean', 'sum', 'min', 'max'])
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?',
                            const=True, default=False)
        parser.add_argument('--exog_size', type=int, default=None)


        return parser


