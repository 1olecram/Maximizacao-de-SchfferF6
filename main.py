import random
import math

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
    return 0.5 + (numerator/denominator)

def generate_pop(tamanho, min_val=-100.0, max_val=100.0):
    """Gera uma população de pares de números reais (x, y)."""
    return [(random.uniform(min_val, max_val), random.uniform(min_val, max_val)) for _ in range(tamanho)]

def main():
    pop = generate_pop(5)
    print(pop)
    print([schafferF6_max(ind[0], ind[1]) for ind in pop])
    print(rank_selection(pop,2))

if __name__ == "__main__":
    main()
