import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Força o uso de uma interface interativa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def graphic_definition():
    """Configuração do gráfico, seu sistema de coordenadas e cores para o fitness"""
    x_space = np.linspace(-10, 10, 400)
    y_space = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x_space, y_space)
    Z = schafferF6(X, Y)
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contourf(X,Y,Z, levels=30,cmap='viridis')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Otimização da função Schaffer’s f6")
    
    # Prepara os pontos da população (vazio por enquanto)
    scatter = ax.scatter([], [], color='red', s=15, edgecolor='black')
    
    # Exibe a janela do gráfico na tela
    plt.show()
    
    return fig, ax, scatter


def rank_selection(population, num_selections):
    """
    Seleciona indivíduos baseados em sua posição (ranking) de fitness.
    A probabilidade de seleção é proporcional ao rank, não à fitness absoluta.
    """
    avaliados = [(ind, schafferF6(ind[0], ind[1])) for ind in population]
    avaliados.sort(key=lambda x: x[1])
    
    individuos_ordenados = []
    for ind, _ in avaliados:
        individuos_ordenados.append(ind)
    
    # Cria os pesos do ranking: 1 para o pior até N para o melhor
    weights_rank = list(range(1, len(population) + 1))
    
    # Realiza a seleção com as probabilidades baseadas nos pesos do ranking
    return random.choices(individuos_ordenados, weights=weights_rank, k=num_selections)

def schafferF6(x, y):
    """Implementação da função Schaffer’s F6 (compatível com escalares e matrizes)."""
    num = np.sin(np.sqrt(x**2 + y**2))**2 - 0.5
    den = (1 + 0.001*(x**2 + y**2))**2
    return 0.5 - (num / den)

def generate_pop(tamanho, min_val=-100.0, max_val=100.0):
    """Gera uma população de pares de números reais (x, y)."""
    return [(random.uniform(min_val, max_val), random.uniform(min_val, max_val)) for _ in range(tamanho)]

def main():
    # Gerando os indivíduos dentro do limite visível do gráfico cartesiano (-10 a 10)
    pop = generate_pop(5, min_val=-10.0, max_val=10.0)
    print(pop)
    print([schafferF6(ind[0], ind[1]) for ind in pop])
    print(rank_selection(pop,2))
    
    # Testando a visualização do gráfico
    graphic_definition()


# Parte de Visualização

#Parte para fazer o gráfico da evolução do fitness

# plt.figure(figsize=(8, 4))
# plt.plot(historico_fitness,color = 'blue')
# plt.title('Evolução do Melhor Fitness')
# plt.xlabel('Geração')
# plt.ylabel('Fitness')
# plt.grid(True)
# plt.savefig('evolucao_fitness.png')
# plt.close()


#Função de atualização para a animação
#Tem algumas variáveis não declaradas aqui, mas é porque elas vão vir da parte da execução

# def update(frame):
#     dados_geracao = historico_populacao[frame]
#     scatter.set_offsets(dados_geracao)
#     ax.set_title(f"Geração {frame} - Melhor Fitness: {historico_fitness[frame]:.4f}")
#     return scatter,

# ani = FuncAnimation(fig, update, frames=NUM_GERACOES, blit=True)
# ani.save('evolucao.gif', writer=PillowWriter(fps=10))
if __name__ == "__main__":
    main()
