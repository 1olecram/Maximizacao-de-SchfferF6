import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Força o uso de um backend sem interface gráfica para salvar arquivos
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


def tournament_selection(population, num_selections, tournament_size=3):
    """
    Seleciona indivíduos usando torneio. Mantém uma melhor diversidade na população.
    """
    selected = []
    for _ in range(num_selections):
        tournament = random.sample(population, tournament_size)
        best = max(tournament, key=lambda ind: schafferF6(ind[0], ind[1]))
        selected.append(best)
    return selected

def schafferF6(x, y):
    """Implementação da função Schaffer’s F6 (compatível com escalares e matrizes)."""
    num = np.sin(np.sqrt(x**2 + y**2))**2 - 0.5
    den = (1 + 0.001*(x**2 + y**2))**2
    return 0.5 - (num / den)

def generate_pop(tamanho, min_val=-100.0, max_val=100.0):
    """Gera uma população de pares de números reais (x, y)."""
    return [(random.uniform(min_val, max_val), random.uniform(min_val, max_val)) for _ in range(tamanho)]

def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    """Aplica o crossover BLX-alpha com um distanciamento mínimo para não colapsar."""
    min_x = min(parent1[0], parent2[0])
    max_x = max(parent1[0], parent2[0])
    range_x = max((max_x - min_x), 0.05) # Impede o colapso do delta
    c1_x = random.uniform(min_x - alpha * range_x, max_x + alpha * range_x)
    c2_x = random.uniform(min_x - alpha * range_x, max_x + alpha * range_x)
    
    min_y = min(parent1[1], parent2[1])
    max_y = max(parent1[1], parent2[1])
    range_y = max((max_y - min_y), 0.05)
    c1_y = random.uniform(min_y - alpha * range_y, max_y + alpha * range_y)
    c2_y = random.uniform(min_y - alpha * range_y, max_y + alpha * range_y)
    
    return (c1_x, c1_y), (c2_x, c2_y)

def gaussian_mutation(individual, mutation_rate, min_val, max_val, mu=0, sigma=1.0):
    """
    Aplica mutação gaussiana a um indivíduo com probabilidade mutation_rate.
    Também inclui mutação uniforme (com 10% de chance dentro da mutação) para varrer todo o espaço.
    """
    mutated_x, mutated_y = individual[0], individual[1]
    
    if random.random() < mutation_rate:
        if random.random() < 0.15:
            mutated_x = random.uniform(min_val, max_val)
        else:
            mutated_x += random.gauss(mu, sigma)
            mutated_x = max(min_val, min(mutated_x, max_val)) 
   
    if random.random() < mutation_rate:
        if random.random() < 0.15:
            mutated_y = random.uniform(min_val, max_val)
        else:
            mutated_y += random.gauss(mu, sigma)
            mutated_y = max(min_val, min(mutated_y, max_val))
        
    return (mutated_x, mutated_y)

def main():
    # --- Hiperparâmetros do Algoritmo Genético ---
    min_val = -10.0
    max_val = 10.0
    pop_size = 300  # População maior para maior diversidade e evitar ótimos locais
    mutation_rate = 0.2  # 20% de chance de mutação por gene
    mutation_sigma_initial = 1.0 # Sigma inicial alto para exploração
    mutation_sigma_final = 0.001 # Sigma final muito baixo para fine-tuning
    max_generations = 300 # Aumentado para dar tempo para explorar os ótimos locais
    tolerance = 1e-6 # Fator de término mais rigoroso para não parar à toa
    
    # --- Geração Inicial ---
    pop = generate_pop(pop_size, min_val, max_val)
    
    prev_best_fitness = -float('inf')
    historic_pop = []
    historic_fitness = []
    stagnation_counter = 0
    max_stagnation = 30 # Para se o melhor não melhorar por 30 gerações sucessivas

    for geracao in range(max_generations):
        # Sigma decai linearmente ao longo das gerações
        current_mutation_sigma = mutation_sigma_initial - (mutation_sigma_initial - mutation_sigma_final) * (geracao / max_generations)

        # Avalia a população atual
        fitnesses = [schafferF6(ind[0], ind[1]) for ind in pop]
        current_avg_fitness = sum(fitnesses) / pop_size
        best_fitness = max(fitnesses)

        # Armazena a "Foto" da geração atual
        historic_pop.append(pop.copy())  
        historic_fitness.append(best_fitness)

        print(f"Geração {geracao:03d} | Melhor Fitness: {best_fitness:.6f} | Média: {current_avg_fitness:.6f}")
        
        # Critério de Parada: atingiu o máximo global
        if abs(1.0 - best_fitness) < tolerance:
            print(f"\nAlgoritmo convergiu no máximo global na geração {geracao}!")
            break
        
        # --- Seleção ---
        selected_parents = tournament_selection(pop, pop_size, tournament_size=3)
        
        next_generation = []
        
        # --- Elitismo e Busca Local (Algoritmo Memético) ---
        best_ind = max(pop, key=lambda ind: schafferF6(ind[0], ind[1]))
        
        # Faz uma micro-busca local no entorno do melhor indivíduo
        # Isso garante que ele suba o pico exato caso pouse perto (fine-tuning extremo)
        for _ in range(10):
            test_x = best_ind[0] + random.gauss(0, current_mutation_sigma * 0.5)
            test_y = best_ind[1] + random.gauss(0, current_mutation_sigma * 0.5)
            best_fitness_val = schafferF6(best_ind[0], best_ind[1])
            test_fitness_val = schafferF6(test_x, test_y)
            if test_fitness_val > best_fitness_val:
                best_ind = (test_x, test_y)
                
        next_generation.append(best_ind)
        
        # --- Crossover ---
        for i in range(0, pop_size - 1, 2):
            p1 = selected_parents[i]
            if i + 1 < pop_size - 1:
                p2 = selected_parents[i+1]
                child1, child2 = blx_alpha_crossover(p1, p2, alpha=0.5)
                next_generation.extend([child1, child2])
            else:
                next_generation.append(p1)
                
        # --- Mutação ---
        # Mutamos todos, EXCETO o indivíduo elite (índice 0)
        mutated_children = [gaussian_mutation(child, mutation_rate, min_val, max_val, sigma=current_mutation_sigma) for child in next_generation[1:]]
        next_generation = [next_generation[0]] + mutated_children
                
        # A nova geração substitui a anterior para o próximo ciclo
        pop = next_generation
    
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
