import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DDPG
from control_system import Sys
import time
import random
from transformer_feature_extractor import TransformerFeatureExtractor


def train():
	env = Sys(num_clouds=1, num_edges=2, num_clients=10, verbose=False, action_discrete=False)
	# check_env(env)
	# print("env checked!", flush=True)

	# print("Start training!", flush=True)

	start_time = time.time()


	# model = PPO("MlpPolicy", env, n_steps=128, verbose=1, tensorboard_log="logs/ppo/", device="cpu")
	# model = DDPG(
	# 	"MlpPolicy",
	# 	env,
	# 	buffer_size=10000,
	# 	learning_starts=0,
	# 	train_freq=(128, "step"),
	# 	gradient_steps=1,
	# 	batch_size=64,
	# 	verbose=1,
	# 	tensorboard_log="logs/ddpg",
	# )

	# # model = DDPG(
	# # 	"MlpPolicy",
	# # 	env,
	# # 	buffer_size=100,
	# # 	learning_starts=0,
	# # 	train_freq=(3, "step"),
	# # 	gradient_steps=3,
	# # 	batch_size=2,
	# # 	verbose=1,
		
	# # )

	# model.learn(total_timesteps=20000, log_interval=1)
	# model.save("saved_models/ddpg_g_step=1")
	# print("Training done!", flush=True)

	# end_time = time.time()
	# t = end_time - start_time
	# print(f"Total time: {t}", flush=True)






	start_time = time.time()

	# env = DummyVecEnv([lambda: MyEnv()])

	policy_kwargs = dict(
		features_extractor_class=TransformerFeatureExtractor,
		features_extractor_kwargs=dict(features_dim=64, d_model=32, nhead=2, num_layers=2),
		net_arch=dict(pi=[64, 64], vf=[64, 64])
	)


	model = PPO("MlpPolicy",
		env,
		n_steps=128,
		verbose=1,
		tensorboard_log="logs/tppo/",
		device="cuda",
		policy_kwargs=policy_kwargs,
	)
	model.learn(total_timesteps=20000, log_interval=1)
	model.save("saved_models/tppo2")
	print("Training done!", flush=True)

	end_time = time.time()
	t = end_time - start_time
	print(f"Total time: {t}", flush=True)














	# start_time = time.time()

	# # env = DummyVecEnv([lambda: MyEnv()])

	# policy_kwargs = dict(
	# 	features_extractor_class=TransformerFeatureExtractor,
	# 	features_extractor_kwargs=dict(features_dim=64, d_model=32, nhead=2, num_layers=2),
	# 	net_arch=dict(pi=[64, 64], qf=[64, 64])
	# )



	# model = DDPG(
	# 	"MlpPolicy",
	# 	env,
	# 	buffer_size=10000,
	# 	learning_starts=0,
	# 	train_freq=(128, "step"),
	# 	gradient_steps=4,
	# 	batch_size=64,
	# 	verbose=1,
	# 	tensorboard_log="logs/tddpg",
	# 	policy_kwargs=policy_kwargs,
	# 	device="cuda",
	# )


	# model.learn(total_timesteps=20000, log_interval=1)
	# model.save("saved_models/tddpg")
	# print("Training done!", flush=True)

	# end_time = time.time()
	# t = end_time - start_time
	# print(f"Total time: {t}", flush=True)





def eval_model():

	env = Sys(num_clouds=1, num_edges=2, num_clients=10, verbose=False, action_discrete=True)
	# model = DDPG.load("trained_agent/DDPG_g_step=4/ddpg.zip", env=env, device="cpu")
	# model = PPO.load("trained_agent/PPO_20000_0/ppo.zip", env=env, device="cuda")
	model = PPO.load("trained_agent/TPPO/tppo2.zip", env=env, device="cpu")
	

	start_time = time.time()

	obs, _ = env.reset()
	total_reward = 0.0

	action_time = 0

	for _ in range(128):
		# action, _states = model.predict(obs, deterministic=True)
		# action = 0
		
		# action = random.randint(0,2)

		# action = [random.random(), random.random(), random.random(), random.random()]

		action_start = time.time()

		action, _states = model.predict(obs, deterministic=True)

		action_time += time.time() - action_start

		obs, reward, done, _, _ = env.step(action)
		total_reward += reward
		if done:
			print(f"done 1 itr: {time.time() - start_time}, reward: {total_reward}, avg action time: {action_time/128.0}")
			obs = env.close()

	end_time = time.time()
	t = end_time - start_time
	print(f"Total time: {t}", flush=True)
