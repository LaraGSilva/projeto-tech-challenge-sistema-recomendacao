# projeto-tech-challenge-sistema-recomendacao
Repositório destinado ao segundo projeto da pós tech. É um sistema de recomendação desenvolvido com um mlp e pytorch, além de boas praticas clean code.



- representação de dadosem sistemas de recomendaçãp: tabela atributo-valor
- Objetivo: predizer quanto um usuario vai gostar de um determinado item.
- passo a passos:
    - filtragem colaborativa: 
        - calcular a similaridade entre o usuario alvo e todos os usuarios do sistema. COnsiderar os itens em comum que os usuairos avaliaram/compraram
        - selecionar os k usuarios mais similares
        - calcular a media das avaliações que os usuarios similares deram para o item em questao

    - filtragem por conteudo: 
        - calcular a similaridade entre o item-alvo da recomendação e todos os outros itedens
        - selecionar os k itens mais similares para os quais o usuario ja fez avaliações
        - calcialr a media das avaliações pelo usuario aos k itens 


- para o modelo de sistema de recomendação vou utilziar o algoritmo de filtragem por conteudo, pois iremos criar uma predição com base nas caracteristicas do produto.


- dados origens: registros de interação no site: view, click, transação
