def train_ddqn(env, agent, episodes=1000, epsilon_decay=0.995, min_epsilon=0.01, target_update_frequency=10):
    rewards_history = []
    epsilon = 1.0

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state

            # Update the network weights with a batch of samples from the experience replay
            agent.update()

        # Decay the exploration rate epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update target network periodically
        if episode % target_update_frequency == 0:
            agent.update_target_network()

        rewards_history.append(episode_reward)
        print(f"Episode {episode}/{episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")

    return rewards_history

def test_ddqn(env, agent, num_episodes=10):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # Use epsilon=0 for greedy policy
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        print(f"Test Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
