import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def executar(episodios, em_treinamento=True, renderizar=False):
    ambiente = gym.make('Taxi-v3', render_mode='human' if renderizar else None)

    if em_treinamento:
        q = np.zeros((ambiente.observation_space.n, ambiente.action_space.n))
    else:
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    taxa_aprendizado_a = 0.9
    fator_desconto_g = 0.9
    epsilon = 1
    taxa_decay_epsilon = 0.0001
    rng = np.random.default_rng()

    recompensas_por_episodio = np.zeros(episodios)

    for i in range(episodios):
        estado = ambiente.reset()[0]
        finalizado = False
        truncado = False

        recompensas = 0

        while not finalizado and not truncado:
            if em_treinamento and rng.random() < epsilon:
                acao = ambiente.action_space.sample()
            else:
                acao = np.argmax(q[estado, :])

            novo_estado, recompensa, finalizado, truncado, _ = ambiente.step(acao)

            recompensas += recompensa

            if em_treinamento:
                q[estado, acao] = q[estado, acao] + taxa_aprendizado_a * (
                    recompensa + fator_desconto_g * np.max(q[novo_estado, :]) - q[estado, acao]
                )

            estado = novo_estado

        epsilon = max(epsilon - taxa_decay_epsilon, 0)

        if epsilon == 0:
            taxa_aprendizado_a = 0.0001

        recompensas_por_episodio[i] = recompensas

    ambiente.close()

    soma_recompensas = np.zeros(episodios)
    for t in range(episodios):
        soma_recompensas[t] = np.sum(recompensas_por_episodio[max(0, t-100):(t+1)])
    plt.plot(soma_recompensas)
    plt.savefig('taxi.png')

    if em_treinamento:
        f = open("taxi.pkl", "wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    executar(15000)
    executar(10, em_treinamento=False, renderizar=True)
