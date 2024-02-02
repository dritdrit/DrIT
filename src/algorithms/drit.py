import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class DrIT(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.a_alpha = args.a_alpha
		self.b_beta = args.b_beta
		self.g_gamma = args.g_gamma
		self.if_DrE = args.if_DrE

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None, obs_2=None, action_2=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)    
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)
			# -------------------------------------------------------------------
			obs_ori = obs
			action_ori = action
			# -------------------------------------------------------------------

		if self.a_alpha == self.b_beta:
			obs = utils.cat(obs, augmentations.random_overlay(obs.clone()))   
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)    # currrent_Q.shape:256*1，前128维为obs与a计算，后128维为obs_aug与a计算
			critic_loss = (self.a_alpha + self.b_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))    # 原obs与
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.a_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug_1 = augmentations.random_conv(obs.clone())
			current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_aug_1, action)
			critic_loss += self.b_beta * \
				(F.mse_loss(current_Q1_aug_1, target_Q) + F.mse_loss(current_Q2_aug_1, target_Q))
			
			obs_aug_2 = augmentations.random_overlay(obs.clone())
			current_Q1_aug_2, current_Q2_aug_2 = self.critic(obs_aug_2, action)
			critic_loss += self.g_gamma * \
				(F.mse_loss(current_Q1_aug_2, target_Q) + F.mse_loss(current_Q2_aug_2, target_Q))


		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
	
		# --------------------------------------------------------------------
		obs_2 = torch.cat((obs_ori, obs_2), dim=0)
		action_2 = torch.cat((action_ori, action_2), dim=0)

		obs_2_aug_1 = augmentations.random_conv(obs_2.clone())
		obs_2_aug_2 = augmentations.random_overlay(obs_2.clone())

		current_Q1_ori, current_Q2_ori = self.critic(obs_2, action_2)
		current_Q1_ori_aug_1, current_Q2_ori_aug_1 = self.critic(obs_2_aug_1, action_2)
		current_Q1_ori_aug_2, current_Q2_ori_aug_2 = self.critic(obs_2_aug_2, action_2)
		loss_critic_encoder = (F.mse_loss(current_Q1_ori.detach(), current_Q1_ori_aug_1) + F.mse_loss(current_Q2_ori.detach(), current_Q2_ori_aug_1)
						 + F.mse_loss(current_Q1_ori.detach(), current_Q1_ori_aug_2) + F.mse_loss(current_Q2_ori.detach(), current_Q2_ori_aug_2)) * 0.1
	
		self.critic_encoder_optimizer.zero_grad()
		loss_critic_encoder.backward()
		self.critic_encoder_optimizer.step()

		return obs_2_aug_1, obs_2_aug_2
		# --------------------------------------------------------------------
	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()    
		obs_2, action_2 = replay_buffer.sample_encoder()
		obs_2_aug_1, obs_2_aug_2 = self.update_critic(obs, action, reward, next_obs, not_done, L, step, obs_2, action_2)



		if step % self.actor_update_freq == 0:

			if not self.if_DrE:
				obs_2 = torch.cat((obs, obs_2), dim=0)
				mu_ori, _, _, log_std_ori = self.actor(obs,compute_pi=False, compute_log_pi=False)

				mu_1, _, _, log_std_1 = self.actor(obs_2_aug_1,compute_pi=False, compute_log_pi=False)
				mu_2, _, _, log_std_2 = self.actor(obs_2_aug_2,compute_pi=False, compute_log_pi=False)

				kl_divergence_ori_aug_1 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_1, log_std_1)
				kl_divergence_ori_aug_2 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_2, log_std_2)

				kl_loss = (kl_divergence_ori_aug_1 + kl_divergence_ori_aug_2).mean()
				betta = 0.1
	
				obs_aug = torch.cat((obs_2, obs_2_aug_1, obs_2_aug_2), dim=0)

				self.update_actor_and_alpha(obs_aug, L, step, kl_loss=kl_loss*betta)
			else:
				self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

	def kl_divergence(self, mu_p, p_log_std, mu_q, q_log_std):
		p_std = torch.exp(p_log_std)
		q_std = torch.exp(q_log_std)


		kl = torch.log(q_std / p_std) + (p_std**2 + (mu_p - mu_q)**2) / (2 * q_std**2) - 0.5
		kl = torch.mean(kl, dim=-1)
		return kl
