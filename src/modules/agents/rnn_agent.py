import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, name, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.name = name

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(input_shape, 4)

        self.move_fc = nn.Linear(args.move_feats_size + args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.enemy_fc = nn.Linear(args.enemy_feats_size + args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.ally_fc = nn.Linear(args.ally_feats_size + args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.own_fc = nn.Linear(args.own_feats_size + args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc_q = nn.Linear(4 + args.rnn_hidden_dim, args.n_actions)
        
        
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state,  macro_inputs, select_action = False):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q_weights_emb = F.softmax(self.fc2(inputs), -1)

        move_emb = self.move_fc(th.cat([macro_inputs[:,0,:self.args.move_feats_size], h], -1))
        enemy_emb = self.enemy_fc(th.cat([macro_inputs[:,1,:self.args.enemy_feats_size], h], -1))
        ally_emb = self.ally_fc(th.cat([macro_inputs[:,2,:self.args.ally_feats_size], h], -1))
        own_emb = self.own_fc(th.cat([macro_inputs[:,3,:self.args.own_feats_size], h], -1))

        move_q = self.fc_q(th.cat([move_emb, q_weights_emb], -1))
        enemy_q = self.fc_q(th.cat([enemy_emb, q_weights_emb], -1))
        ally_q = self.fc_q(th.cat([ally_emb, q_weights_emb], -1))
        own_q = self.fc_q(th.cat([own_emb, q_weights_emb], -1))
        
        # options
        # q = th.stack([move_q, enemy_q, ally_q, own_q], 1).gather(1, q_weights_emb.argmax(-1, keepdim=True).unsqueeze(-1).repeat(1,1,own_q.shape[-1])).squeeze(1)

        # q_weights_emb = F.softmax(self.fc2(th.cat([move_emb, enemy_emb, ally_emb, own_emb], -1)), -1)
        # regular_q = th.stack([move_q, enemy_q, ally_q, own_q], 1).sort(1, True)[0]
        # print(regular_q.shape, q_weights_emb.shape)
        
        q = q_weights_emb.unsqueeze(-1).repeat(1, 1, self.args.n_actions) * th.stack([move_q, enemy_q, ally_q, own_q], 1) 
        q = q.sum(1)
        
        # select action 为后面分析时可视化权重使用，训练时用不到
        if select_action:
            return q, h, q_weights_emb.detach().cpu().numpy()
        else:
            return q, h


class Multihead_Module(nn.Module):
    def __init__(self, args):
        super(Multihead_Module, self).__init__()
        self.input_size, self.num_heads, self.embedding_size = args

        self.embedding_q = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        self.embedding_k = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        self.embedding_v = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        self.multihead_net = nn.MultiheadAttention(self.embedding_size, self.num_heads)

    def forward(self, inputs):
        q_vec = self.embedding_q(inputs).permute(1,0,2)
        k_vec = self.embedding_k(inputs).permute(1,0,2)
        v_vec = self.embedding_v(inputs).permute(1,0,2)
        multihead_op, multihead_weights = self.multihead_net(q_vec, k_vec, v_vec)
        return F.softmax(multihead_op.permute(1,0,2), dim = -2)

if __name__ == "__main__":
    multi_head_module = Multihead_Module(args = (32, 1, 1))
    test_inputs = th.randn((4, 3, 32))
    op = multi_head_module(test_inputs)
    p = (op.squeeze(-1).unsqueeze(1) * th.randn((3, 6, 4))).sum(-1)
    print(op, op.shape, p, p.shape)