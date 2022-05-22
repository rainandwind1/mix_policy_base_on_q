import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.graph_qmix import Graph_QMixer
from modules.mixers.multi_head_qmix import Multihead_QMixer
from modules.mixers.multihead_multifeats_qmix import Multihead_multifeats_QMixer
from modules.mixers.multi_graph_qmix import Multi_Graph_QMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, name, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.name = name

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if self.name is not None:
            if args.mixer is not None:
                if args.mixer == "vdn":
                    self.mixer = VDNMixer()
                elif args.mixer == "qmix":
                    self.mixer = QMixer(args)
                elif args.mixer == "graph_qmix":
                    self.mixer = Graph_QMixer(args, scheme)
                elif args.mixer == "multi_head_qmix":
                    self.mixer = Multihead_QMixer(args, scheme)
                elif args.mixer == "multihead_multifeats_qmix":
                    self.mixer = Multihead_multifeats_QMixer(args, scheme)
                elif args.mixer == "multi_graph_qmix":
                    self.mixer = Multi_Graph_QMixer(args, scheme)
                else:
                    raise ValueError("Mixer {} not recognised.".format(args.mixer))
                self.params += list(self.mixer.parameters())
                self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            # weight_entropy_out.append(weight_entropy)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # weight_entropy_out = th.stack(weight_entropy_out, dim=1).view(batch.batch_size, batch.max_seq_length, -1)
        # print(mac_out.shape, weight_entropy_out.shape)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # chosen_action_probs = th.gather(action_probs_out[:, :-1], dim=3, index=actions).squeeze(3)
        # print(chosen_action_qvals.shape, chosen_action_probs.shape)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.name is not None:
            if self.mixer is not None:
                if self.args.mixer == "graph_qmix":
                    chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1], batch["adjacency_matrix"][:, :-1])
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["obs"][:, 1:], batch["adjacency_matrix"][:, 1:])
                elif self.args.mixer == "multi_head_qmix":
                    chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1])
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["obs"][:, 1:])
                elif self.args.mixer == "multihead_multifeats_qmix":
                    feats_ls = (batch["move_feats"][:,:-1], batch["enemy_feats"][:, :-1], batch["ally_feats"][:, :-1], batch["own_feats"][:, :-1])
                    next_feats_ls = (batch["move_feats"][:, 1:], batch["enemy_feats"][:, 1:], batch["ally_feats"][:, 1:], batch["own_feats"][:, 1:])
                    chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], feats_ls)
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], next_feats_ls)
                elif self.args.mixer == "multi_graph_qmix":
                    feats_ls = (batch["move_feats"][:,:-1], batch["enemy_feats"][:, :-1], batch["ally_feats"][:, :-1], batch["own_feats"][:, :-1])
                    next_feats_ls = (batch["move_feats"][:, 1:], batch["enemy_feats"][:, 1:], batch["ally_feats"][:, 1:], batch["own_feats"][:, 1:])
                    chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], feats_ls, batch["adjacency_matrix"][:, :-1])
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], next_feats_ls, batch["adjacency_matrix"][:, 1:])
                else:
                    chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        # if self.name == "micro policy":
        #     rewards = rewards.squeeze(-1)
            # print(rewards.shape, target_max_qvals.shape)
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        # print(td_error.shape, chosen_action_qvals.shape, weight_entropy_out.shape)
        # td_error -= weight_entropy_out.sum(-1, keepdim=True)[:,:-1,:]

        # actor_loss = -th.log(chosen_action_probs.mean(-1, keepdim=True)) * chosen_action_qvals.detach()
        # td_error += actor_loss

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(self.name + "_loss", loss.item(), t_env)
            self.logger.log_stat(self.name + "_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(self.name + "_td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat(self.name + "_q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat(self.name + "_target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/".format(path) + self.name + "_mixer.th")
        th.save(self.optimiser.state_dict(), "{}/".format(path) + self.name + "_opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/".format(path) + self.name + "_mixer.th", map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/".format(path) + self.name + "_opt.th", map_location=lambda storage, loc: storage))
