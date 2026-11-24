from selfplay.self_play import SelfPlay, random_policy


if __name__ == "__main__":
    from experiments.connect4 import make_env

    env = make_env()
    try:
        current, opponent = SelfPlay().play_a_game(env, random_policy, random_policy)
        print("Current policy reward:", sum(current.reward))
        print("Opponent policy reward:", sum(opponent.reward))
    finally:
        env.close()
