# Maximizacao de Schffer F6

Maximizar a função F6 de Schaffer testa se um algoritmo consegue explorar o espaço de busca (diversidade) e, ao mesmo tempo, convergir para a melhor solução (intensidade), superando armadilhas locais.

## Bibliotecas do projeto
* random
* numpy
* matplotlib

## Como executar

**Ambiente virtual python**  
* No Windows:
    ```
    cd caminho\para\seu\projeto
    python -m venv venv
    .\venv\Scripts\Activate.ps1 (powershell)
    .\venv\Scripts\activate.bat (cmd)
    ```
  
* No Linux:
    ```
    cd caminho\para\seu\projeto
    python3 -m venv venv
    source venv/bin/activate
    ```

* Para desativar:
    ```
    deactivate
    ```

**Requirements.txt**
* Dentro do ambiente virtual, execute:
    ```
    pip install -r requirements.txt
    ```
* Caso novo pacote seja adicionado, execute:
    ```
    pip freeze > requirements.txt 
    ```
## Como manipular
**Algoritmo genético** 
* Valores de precisão podem ser alterados para variar a convergência do algoritmo:

1. Altere `pop_size` para mudar o tamanho da população
2. Altere `mutation_rate` para mudar a taxa de mutação
3. Altere `mutation_sigma` para mudar o desvio padrão da mutação
4. Altere ` max_generations` para mudar o número máximo de gerações 
5. Altere `tolerance` para mudar a variação mínima aceitável do fitness médio
    
**Parte gráfica** 

* Valores que podem ser alterados para mudar a aparência do mapa

1. Altere `min_val` e `max_val` no `main` para expandir ou reduzir a área visível do gráfico
2. Altere `levels` dentro de `ax.contourf` para aumentar ou diminuir a quantidade de "camadas" de cor que formam o mapa de fundo
3. Altere `cmap` dentro de `ax.contourf` para mudar a paleta de cores do gráfico de contorno.
4. Altere a variável `s` (size) dentro de `ax.scatter` para deixar as bolinhas que representam os indivíduos maiores ou menores na tela
5. Altere `interval` dentro de `FuncAnimation` para mudar a velocidade em que a animação roda na janela do programa
6. Altere `fps` dentro de `PillowWriter` para deixar o arquivo de GIF exportado mais rápido ou mais lento