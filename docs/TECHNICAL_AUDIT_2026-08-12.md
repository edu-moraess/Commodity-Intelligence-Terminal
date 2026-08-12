# Auditoria Técnica e Validação de Correção

**Projeto:** Commodity Intelligence Terminal  
**Data:** 12 de agosto de 2026  
**Base auditada:** `main` em `eab0835` antes desta alteração

## Escopo e método

Esta revisão registrou o estado do repositório clonado, percorreu a estrutura de módulos Python, verificou o fluxo central de dados, identificou usos de geração sintética e executou a suíte automatizada. A validação inicial revelou uma quebra de compatibilidade na API pública de correlação: o teste que chama `correlation_matrix(panel)` não conseguia executar porque `window` era obrigatório. A implementação agora define uma janela anual de **252 pregões** como padrão, sem alterar chamadas existentes que fornecem uma janela explícita.[1] [2]

> A execução de dados externos em produção não foi usada como critério de validação desta revisão. A disponibilidade de Yahoo Finance, FRED e de credenciais permanece dependente do ambiente de execução.

## Matriz de auditoria focalizada

| Componente | Estado atual | Achado principal | Ação | Prioridade |
|---|---|---|---|---|
| Camada de dados | Centralizada e reutilizável | `DataManager` concentra preços e macro, faz download em lote e aplica cache por tipo de dado. | **KEEP** | P1 |
| API de correlação | Interface inconsistente | A assinatura exigia `window`, enquanto o contrato de teste também suporta a chamada sem esse argumento. | **REFACTOR** — concluído neste commit | P0 |
| Fontes de mercado e macro | Funcionais, com dependências externas | Yahoo Finance e FRED são adaptadores atuais; falhas são tratadas no fluxo de dados. | **KEEP** com monitoramento | P1 |
| Fallback sintético | Transparente, mas metodologicamente limitado | Séries sintéticas são identificadas por `is_synthetic` e pela origem, porém ainda podem ocupar o lugar de dados indisponíveis. | **REFACTOR** — incluir modo estrito que exiba `Data unavailable` | P1 |
| Forecasting e risco | Cobertos por testes automatizados | A suíte verifica cenários, Monte Carlo, validação probabilística, risco, regimes e cache; ela não certifica desempenho fora da amostra em produção. | **KEEP** com validação contínua | P2 |
| Arquitetura institucional alvo | Parcialmente implementada | O projeto possui módulos de dados, analytics, forecast, risco, portfólio e testes, mas não possui ainda a árvore completa de `src/`, proveniência canônica, registry de modelos e documentação metodológica separada. | **REBUILD** incremental, orientado por dados e credenciais disponíveis | P2 |
| Dependências | Declaradas e reproduzíveis | `requirements.txt` limita versões principais e separa dependências avançadas ainda não usadas. | **KEEP** | P2 |

## Alteração realizada

A função `analytics.correlation.correlation_matrix` passou a aceitar `window: int = 252`. O valor representa aproximadamente um ano de pregões e preserva o comportamento anterior para os consumidores que já fornecem janelas específicas, como 126 ou 252 pregões. Para painéis menores, a função continua limitando a janela ao número de observações disponível.[2]

| Arquivo | Alteração | Efeito esperado |
|---|---|---|
| `analytics/correlation.py` | Inclusão da janela padrão e atualização da docstring pública. | Restaura compatibilidade com chamadas sem `window` e elimina a falha de regressão observada. |
| `docs/TECHNICAL_AUDIT_2026-08-12.md` | Inclusão deste registro de auditoria. | Mantém os achados, a decisão e os limites conhecidos rastreáveis no repositório. |

## Validação executada

A suíte foi executada após a correção com `pytest tests/ -q --disable-warnings --maxfail=1`.

| Métrica | Resultado |
|---|---:|
| Testes aprovados | **79** |
| Testes com falha | **0** |
| Tempo de execução | **13,40 s** |
| Verificação da regressão de correlação | **Aprovada** |

## Dependências e próximos incrementos

A aplicação usa Yahoo Finance para preços e FRED para macroeconomia, ambos encapsulados pela camada de dados atual.[1] Para seguir integralmente a arquitetura institucional pretendida, os próximos incrementos devem priorizar um modo sem fallback sintético para uso analítico, metadados de proveniência por observação, providers adicionais para fundamentos e curva de futuros, registry de modelos e documentação metodológica por domínio. Essas evoluções exigem fontes de dados, credenciais e validação histórica adicionais; elas não devem ser simuladas como se fossem dados de produção.[4] [5] [6]

## Referências

[1]: ../data/data_manager.py "Camada central de dados"
[2]: ../analytics/correlation.py "Funções de correlação"
[3]: ../tests/test_forecasting_and_correlation.py "Contrato automatizado de correlação e forecast"
[4]: ../requirements.txt "Dependências do projeto"
[5]: ../config/settings.py "Configuração de ativos e parâmetros"
[6]: ../README.md "Arquitetura e roadmap atuais"
