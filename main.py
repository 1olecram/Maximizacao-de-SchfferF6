import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Força o uso de uma interface interativa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def graphic_definition(lim_min, lim_max):
    """Configuração do gráfico, seu sistema de coordenadas e cores para o fitness"""
    #criando o ambiente cartesiano do gráfico
    x_space = np.linspace(lim_min, lim_max, 400)
    y_space = np.linspace(lim_min, lim_max, 400)
    X, Y = np.meshgrid(x_space, y_space)
    Z = schafferF6(X, Y)
    
    fig, ax = plt.subplots(figsize=(8, 8)) #criando a janela de visualizacao (8" x 8")
    contour = ax.contourf(X,Y,Z, levels=30,cmap='viridis')
    
    #Limites visuais
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_title("Otimização da função Schaffer’s f6")
    
    scatter = ax.scatter([], [], color='red', s=15, edgecolor='black') # Prepara os pontos da população (vazio por enquanto)
    
    plt.show() # Exibe a janela do gráfico na tela
    
    return fig, ax, scatter, contour


def rank_selection(population, num_selections):
    """
    Seleciona indivíduos baseados em sua posição (ranking) de fitness.
    A probabilidade de seleção é proporcional ao rank, não à fitness absoluta.
    """
    avaliados = [(ind, schafferF6(ind[0], ind[1])) for ind in population]
    avaliados.sort(key=lambda x: x[1])

    # Cria os pesos do ranking: 1 para o pior até N para o melhor
    weights_rank = list(range(1, len(population) + 1))
    
    # Extrai apenas as coordenadas ordenadas (remove o valor do fitness da tupla)
    individuos_ordenados = [ind for ind, fitness in avaliados]
    
    # Realiza a seleção com as probabilidades baseadas nos pesos do ranking
    # Nota: random.choices faz seleção COM reposição por padrão
    return random.choices(individuos_ordenados, weights=weights_rank, k=num_selections)

def schafferF6(x, y):
    """Implementação da função Schaffer’s F6 (compatível com escalares e matrizes)."""
    num = np.sin(np.sqrt(x**2 + y**2))**2 - 0.5
    den = (1 + 0.001*(x**2 + y**2))**2
    return 0.5 - (num / den)

def generate_pop(tamanho, min_val=-100.0, max_val=100.0):
    """Gera uma população de pares de números reais (x, y)."""
    return [(random.uniform(min_val, max_val), random.uniform(min_val, max_val)) for _ in range(tamanho)]

def arithmetic_crossover(parent1, parent2):
    """Aplica o crossover aritmético entre dois pais para gerar dois filhos."""
    alpha = random.random()  # Peso aleatório entre 0.0 e 1.0
    
    # Filho 1: alpha * P1 + (1 - alpha) * P2
    c1_x = alpha * parent1[0] + (1 - alpha) * parent2[0]
    c1_y = alpha * parent1[1] + (1 - alpha) * parent2[1]
    
    # Filho 2: (1 - alpha) * P1 + alpha * P2
    c2_x = (1 - alpha) * parent1[0] + alpha * parent2[0]
    c2_y = (1 - alpha) * parent1[1] + alpha * parent2[1]
    
    return (c1_x, c1_y), (c2_x, c2_y)

def main():
    # Gerando os indivíduos dentro do limite visível do gráfico cartesiano (-10 a 10)
    min_val=-10.0
    max_val=10.0
    pop_size = 5
    pop = generate_pop(pop_size, min_val, max_val)
    
    # Seleciona os pais para a próxima geração (com reposição)
    selected_parents = rank_selection(pop, pop_size)
    
    next_generation = []
    
    # Itera sobre os pais selecionados de 2 em 2
    for i in range(0, pop_size, 2):
        p1 = selected_parents[i]
        # Verifica se existe um par para fazer o crossover (em caso de pop_size ímpar)
        if i + 1 < pop_size:
            p2 = selected_parents[i+1]
            child1, child2 = arithmetic_crossover(p1, p2)
            next_generation.extend([child1, child2])
        else:
            # Se sobrou um indivíduo sem par, ele passa direto para a próxima geração
            next_generation.append(p1)
            
    print("População anterior:", pop)
    print("Nova geração:", next_generation)
    
    # Testando a visualização do gráfico
    graphic_definition(min_val, max_val)


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
