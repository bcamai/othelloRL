import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64)  
        )

    def forward(self, x):
        return self.net(x)
    
def build_adjency_matrix(size=8):
    num_nodes = size*size
    total_nodes = num_nodes + 1 # NOTE this for dummy node that is connected to all other
    
    adj = torch.zeros((total_nodes, total_nodes))

    for row in range(size):
        for col in range(size):
            node_idx = row + size * col

            adj[node_idx, node_idx] = 1.0 # NOTE self loop to wrap it all in one matmul

            directions = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1),          (0, 1),
                          (1, -1), (1, 0), (1, 1)]
            for d_row, d_col in directions:
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < size and 0 <= n_col < size:
                    neighbor_idx = n_row * size + n_col
                    adj[node_idx, neighbor_idx] = 1.0

    dummy_idx = num_nodes
    adj[dummy_idx, dummy_idx] = 1.0

    for i in range(num_nodes):
        adj[i, dummy_idx] = 1.0
        adj[dummy_idx, i] = 1.0

    # NOTE Normalization so gradient doesn't explode, and nodes with more neighbors won't dominate
    #row_sums = adj.sum(dim=1, keepdim=True)
    #adj = adj / row_sums

    return adj

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        support = self.linear(x)
        output = torch.matmul(adj_matrix.unsqueeze(0), support)
        return F.leaky_relu(output, negative_slope=0.1)

class DQN_GNN(nn.Module):
    def __init__(self, board_size=8, hidden_dim=64):
        super(DQN_GNN, self).__init__()

        # NOTE as a buffer for gpu training
        self.register_buffer('adj_matrix', build_adjency_matrix(board_size))
        self.num_nodes = board_size * board_size

        self.gcn1 = GraphConvLayer(in_features=2, out_features=hidden_dim)
        self.gcn2 = GraphConvLayer(in_features=hidden_dim, out_features=hidden_dim)
        self.gcn3 = GraphConvLayer(in_features=hidden_dim, out_features=hidden_dim)

        # NOTE Since in one paper someone said concat helps network taking 
        # into account fine details as well global view hidden_dim is 3 times as large
        self.q_value_head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, 1))

        self.dummy_node_init = nn.Parameter(torch.zeros(1, 1, 2))

    # NOTE for the time being input is tensor of size 128 first 64 are agents pieces next 64 opponents
    def forward(self, state):
        batch_size = state.shape[0]

        agents_pieces = state[:, :self.num_nodes].unsqueeze(-1)
        opponents_pieces = state[:, self.num_nodes:].unsqueeze(-1)
        board_nodes = torch.cat([agents_pieces, opponents_pieces], dim=-1) # [batch, 64, 2]

        dummy_node = self.dummy_node_init.expand(batch_size, -1, -1) # [batch, 1, 2]
        x = torch.cat([board_nodes, dummy_node], dim = 1) #  [batch, 65, 2]

        x1 = self.gcn1(x, self.adj_matrix)
        x2 = self.gcn2(x1, self.adj_matrix)
        x3 = self.gcn3(x2, self.adj_matrix)
        
        # TODO test with and without concat
        x_combined = torch.cat([x1, x2, x3], dim=-1) # [batch, 65, 3 * hidden_dim]

        board_features = x_combined[:, :64, :] # [batch, 64, 3 * hidden_dim]

        q_values = self.q_value_head(board_features) # [batch, 64, 1]
        

class ActorCriticGNN(nn.Module):
    def __init__(self, board_size=8, hidden_dim=64):
        super(ActorCriticGNN, self).__init__()

        self.register_buffer('adj_matrix', build_adjency_matrix(board_size))
        self.num_nodes = board_size * board_size
        # NOTE Projection for testing, if it can help when network gets more information than just their/opponents disc #proj
        self.input_proj = nn.Linear(2, 16)
        self.gcn1 = GraphConvLayer(in_features=16, out_features=hidden_dim)
        self.gcn2 = GraphConvLayer(in_features=hidden_dim, out_features=hidden_dim)
        self.gcn3 = GraphConvLayer(in_features=hidden_dim, out_features=hidden_dim)
        # NOTE 16 dim since other nodes are proj. to 16 dim #proj
        self.dummy_node_init = nn.Parameter(torch.zeros(1, 1, 16))
        
        # NOTE Skipping softmax here, use later after masking legal moves
        self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, 1))

        self.critic_head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, 1),
                nn.Tanh())

    def forward(self, state):
        batch_size = state.shape[0]

        agents_pieces = state[:, :self.num_nodes].unsqueeze(-1) # [batch_size, 64, 1]
        opponents_pieces = state[:, self.num_nodes:].unsqueeze(-1) # [batch_size, 64, 1]

        board_nodes = torch.cat([agents_pieces, opponents_pieces], dim=-1) # [batch_size, 64, 2]
        board_nodes = F.leaky_relu(self.input_proj(board_nodes), 0.1) # [batch_size, 64, 16]

        dummy_node = self.dummy_node_init.expand(batch_size, -1, -1) # [batch_size, 1, 16
        x = torch.cat([board_nodes, dummy_node], dim=1) # [batch_size, 65, 16]

        x1 = self.gcn1(x, self.adj_matrix) # [batch_size, 65, hidden_dim]
        x2 = self.gcn2(x1, self.adj_matrix) # [batch_size, 65, hidden_dim]
        x3 = self.gcn3(x2, self.adj_matrix) # [batch_size, 65, hidden_dim]

        x_combined = torch.cat([x1, x2, x3], dim=-1) # [batch, 65, 3 * hidden_dim]

        board_features = x_combined[:, :self.num_nodes, :] # [batch, 64, 3 * hidden_dim]
        policy_logits = self.actor_head(board_features).squeeze(-1) # [batch, 64]

        virtual_node_features = x_combined[:, self.num_nodes, :] # [batch, 3 * hidden_dim]
        state_value = self.critic_head(virtual_node_features) # [batch, 1]

        return policy_logits, state_value

