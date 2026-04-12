# Maximizacao-de-SchfferF6

## Bibliotecas do projeto
* random
* numpy
* matplotlib

## Como executar

**Ambiente virtual pyhon**  
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