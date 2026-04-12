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
    alpha = random.random()
    
    c1_x = alpha * parent1[0] + (1 - alpha) * parent2[0]
    c1_y = alpha * parent1[1] + (1 - alpha) * parent2[1]
    
    c2_x = (1 - alpha) * parent1[0] + alpha * parent2[0]
    c2_y = (1 - alpha) * parent1[1] + alpha * parent2[1]
    
    return (c1_x, c1_y), (c2_x, c2_y)

def gaussian_mutation(individual, mutation_rate, min_val, max_val, mu=0, sigma=1.0):
    """
    Aplica mutação gaussiana a um indivíduo com uma certa probabilidade.
    Cada gene (x e y) tem uma chance 'mutation_rate' de ser mutado.
    """
    mutated_x, mutated_y = individual[0], individual[1]
    
    # Adicionando uma "macro-mutação" com desvio padrão maior (10% das vezes)
    # Isso permite escapar dos múltiplos ótimos locais da função Schaffer F6
    # que estão separados por aproximadamente 3.14 (pi).
    current_sigma = sigma * 10 if random.random() < 0.1 else sigma
    
    # Possivel mutacao dos genes dentro dos parametros do grafico
    if random.random() < mutation_rate:
        mutated_x += random.gauss(mu, current_sigma)
        mutated_x = max(min_val, min(mutated_x, max_val)) 
   
    if random.random() < mutation_rate:
        mutated_y += random.gauss(mu, current_sigma)
        mutated_y = max(min_val, min(mutated_y, max_val))
        
    return (mutated_x, mutated_y)

def main():
    # --- Hiperparâmetros do Algoritmo Genético ---
    min_val = -10.0
    max_val = 10.0
    pop_size = 150  # População maior para maior diversidade e evitar ótimos locais
    mutation_rate = 0.2  # 20% de chance de mutação por gene
    mutation_sigma = 0.5 # Desvio padrão da mutação
    max_generations = 200 # Aumentado para dar tempo para explorar os ótimos locais
    tolerance = 1e-6 # Fator de término mais rigoroso para não parar à toa
    
    # --- Geração Inicial ---
    pop = generate_pop(pop_size, min_val, max_val)
    
    prev_avg_fitness = -float('inf')
    historic_pop = []
    historic_fitness = []

    for geracao in range(max_generations):
        # Avalia a população atual
        fitnesses = [schafferF6(ind[0], ind[1]) for ind in pop]
        current_avg_fitness = sum(fitnesses) / pop_size
        best_fitness = max(fitnesses)

        # Armazena a "Foto" da geração atual
        historic_pop.append(pop.copy())  
        historic_fitness.append(best_fitness)

        print(f"Geração {geracao:03d} | Melhor Fitness: {best_fitness:.6f} | Média: {current_avg_fitness:.6f}")
        
        # Critério de Parada: Variação de fitness médio
        if abs(current_avg_fitness - prev_avg_fitness) < tolerance:
            print(f"\nAlgoritmo convergiu na geração {geracao}!")
            print(f"A variação da média ({abs(current_avg_fitness - prev_avg_fitness):.8f}) atingiu o fator de término.")
            break
            
        prev_avg_fitness = current_avg_fitness
        
        # --- Seleção ---
        selected_parents = rank_selection(pop, pop_size)
        
        next_generation = []
        
        # --- Elitismo ---
        best_ind = max(pop, key=lambda ind: schafferF6(ind[0], ind[1]))
        next_generation.append(best_ind)
        
        # --- Crossover ---
        for i in range(0, pop_size - 1, 2):
            p1 = selected_parents[i]
            if i + 1 < pop_size - 1:
                p2 = selected_parents[i+1]
                child1, child2 = arithmetic_crossover(p1, p2)
                next_generation.extend([child1, child2])
            else:
                next_generation.append(p1)
                
        # --- Mutação ---
        # Mutamos todos, EXCETO o indivíduo elite (índice 0)
        mutated_children = [gaussian_mutation(child, mutation_rate, min_val, max_val, sigma=mutation_sigma) for child in next_generation[1:]]
        next_generation = [next_generation[0]] + mutated_children
                
        # A nova geração substitui a anterior para o próximo ciclo
        pop = next_generation
    
    # Testando a visualização do gráfico
    graphic_definition(min_val, max_val)
    # Parte de Visualização

    #Gráfico de Linha

    fig_linha, ax_linha = plt.subplots(figsize=(8, 4))
    ax_linha.plot(historic_fitness, color='blue', linewidth=2)
    
    ax_linha.set_title("Evolução do Melhor Fitness")
    ax_linha.set_xlabel("Geração")
    ax_linha.set_ylabel("Fitness (Max = 1.0)")
    ax_linha.grid(True, linestyle='--', alpha=0.7)
    
    # Salva a imagem no seu computador
    fig_linha.savefig("grafico_fitness_schaffer.png", bbox_inches='tight')
    plt.close(fig_linha)
    print("Gráfico 'grafico_fitness_schaffer.png' salvo com sucesso!")

    #Gráfico de Animação
    
    fig, ax , scatter, contour = graphic_definition(min_val, max_val)

    def update(frame):
        # Atualiza os pontos da população no gráfico
        current_pop = historic_pop[frame]
        scatter.set_offsets(current_pop)
        ax.set_title(f"Geração {frame+1} | Melhor Fitness: {historic_fitness[frame]:.6f}")
        return scatter,

    ani = FuncAnimation(fig, update, frames=len(historic_pop), blit=True, repeat=False)
    ani.save("schaffer_f6_optimization.gif", writer=PillowWriter(fps=5))
    

if __name__ == "__main__":
    main()
