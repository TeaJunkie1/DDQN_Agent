from DDQN import DDQNAgent
from COCOObjectDetectionEnv import COCOObjectDetectionEnv
from train_test import train_ddqn,test_ddqn


if __name__ == "__main__":
    env = COCOObjectDetectionEnv('coco/annotations/instances_val2017.json', 'coco/val2017')
    agent = DDQNAgent(state_space=(128, 128, 3), action_space=6)
    # Train the agent
    rewards = train_ddqn(env, agent, episodes=200)
    test_ddqn(env, agent)
