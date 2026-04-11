import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def rank_selection(population, num_selections):
    """
    Seleciona indivíduos baseados em sua posição (ranking) de aptidão.
    A probabilidade de seleção é proporcional ao rank, não à aptidão absoluta.
    """
    avaliados = [(ind, schaﬀerF6_max(ind[0], ind[1])) for ind in population]
    avaliados.sort(key=lambda x: x[1])
    
    individuos_ordenados = []
    for ind, _ in avaliados:
        individuos_ordenados.append(ind)
    
    # Cria os pesos do ranking: 1 para o pior até N para o melhor
    weights_rank = list(range(1, len(population) + 1))
    
    # Realiza a seleção com as probabilidades baseadas nos pesos do ranking
    return random.choices(individuos_ordenados, weights=weights_rank, k=num_selections)

def schaﬀerF6_max(x,y):
    """Implementação da fórmula da função Schaffer’s F 6 para avaliar a qualidade de cada indivíduo (Fitness)."""
    numerator = math.pow(math.sin(math.sqrt((x**2)+(y**2))),2) - 0.5
    denominator = math.pow(1+(0.001*((x**2)+(y**2))),2)
    return 0.5 - (numerator/denominator)

def generate_pop(tamanho, min_val=-100.0, max_val=100.0):
    """Gera uma população de pares de números reais (x, y)."""
    return [(random.uniform(min_val, max_val), random.uniform(min_val, max_val)) for _ in range(tamanho)]

def main():
    pop = generate_pop(5)
    print(pop)
    print([schafferF6_max(ind[0], ind[1]) for ind in pop])
    print(rank_selection(pop,2))


# Parte de Visualização

#Parte para fazer o gráfico da evolução do fitness

plt.figure(figsize=(8, 4))
plt.plot(historico_fitness,color = 'blue')
plt.title('Evolução do Melhor Fitness')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.grid(True)
plt.savefig('evolucao_fitness.png')
plt.close()

#Funcao para criar uma animação dos pontos no gráfico da função

def schaffer_numpy(x,y):
    num = np.sin(np.sqrt(x**2 + y**2))**2 - 0.5
    den = (1 + 0.001*(x**2 + y**2))**2
    return 0.5 - num/den

#Configuração do gráfico para a animação

x_space = np.linspace(-10, 10, 400)
y_space = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x_space, y_space)
Z = schaffer_numpy(X, Y)
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contourf(X,Y,Z, levels=30,cmap='viridis')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("Otimização da função Schaffer’s f6")

scatter = ax.scatter([], [], color='red', s=15, edgecolor='black')

#Função de atualização para a animação
#Tem algumas variáveis não declaradas aqui, mas é porque elas vão vir da parte da execução

def update(frame):
    dados_geracao = historico_populacao[frame]
    scatter.set_offsets(dados_geracao)
    ax.set_title(f"Geração {frame} - Melhor Fitness: {historico_fitness[frame]:.4f}")
    return scatter,

ani = FuncAnimation(fig, update, frames=NUM_GERACOES, blit=True)
ani.save('evolucao.gif', writer=PillowWriter(fps=10))
if __name__ == "__main__":
    main()
