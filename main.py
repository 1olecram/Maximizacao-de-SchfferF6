import random
import math

def schaﬀerF6_max(x,y):
    """Implementação da fórmula da função Schaffer’s F 6 para avaliar a qualidade de cada indivíduo (Fitness)."""
    numerator = math.pow(math.sin(math.sqrt((x**2)+(y**2))),2) - 0.5
    denominator = math.pow(1+(0.001*((x**2)+(y**2))),2)
    return 0.5 + (numerator/denominator)

def generate_pop(tamanho, min_val=-100.0, max_val=100.0):
    """Gera uma população de pares de números reais (x, y)."""
    return [(random.uniform(min_val, max_val), random.uniform(min_val, max_val)) for _ in range(tamanho)]

def main():
    print(schafferF6_max(0.941,86.4598))

if __name__ == "__main__":
    main()
