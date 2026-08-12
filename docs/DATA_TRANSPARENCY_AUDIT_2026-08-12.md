# Auditoria de Procedência e Atualidade dos Dados

**Projeto:** Commodity Intelligence Terminal  
**Data da auditoria:** 12 de agosto de 2026

## Objetivo e método

Esta auditoria verificou a aplicação publicada, a estrutura das abas e a camada de dados do repositório. O foco foi separar três conceitos que anteriormente podiam ser confundidos: o horário de atualização da interface, a última observação disponível em cada série e a natureza da série, que pode ser uma publicação de mercado, um indicador macroeconômico ou uma simulação.

A aplicação publicada carregou com um indicador global de disponibilidade, mas o Dashboard Global informou que todos os ativos mostrados naquele ambiente eram simulados. Por isso, o indicador visual e o carimbo de horário não podiam ser interpretados como confirmação de preços externos atualizados. O comportamento foi corrigido para enfatizar fonte, frequência e data de referência por série.[1] [2]

## Diagnóstico das fontes

A consulta direta ao adaptador Yahoo Finance na data da auditoria retornou uma observação de **12/08/2026** para todos os ativos configurados com essa fonte. Alumínio (`ALI=F`) e minério de ferro (`TIO=F`) estão configurados deliberadamente como séries simuladas, pois não possuem a referência de futuros necessária no provedor atual.[3] [4]

| Grupo | Resultado do diagnóstico | Interpretação correta |
|---|---|---|
| Preços Yahoo Finance | Séries disponíveis na frequência diária, com referência de 12/08/2026 na consulta técnica. | São preços de fechamento diário; não são cotações intradiárias. |
| Aplicação Streamlit publicada | O Dashboard Global exibiu séries simuladas no ambiente observado. | O ambiente publicado precisa de conectividade funcional com o provedor; os valores simulados não representam preços de mercado. |
| Alumínio e minério de ferro | Configurados como sintéticos. | Permanecem explicitamente identificados e não devem ser usados como preço de mercado. |
| Macro FRED | Depende de credencial e do calendário de divulgação do indicador. | Uma referência em julho pode ser normal para uma série mensal, como CPI, PPI ou produção industrial. |

> O botão **Atualizar séries** limpa o cache da aplicação e tenta buscar novamente a fonte. Ele não transforma uma série diária em cotação intradiária nem garante que o provedor externo responderá.

## Matriz das abas

| Aba | Base de dados | Achado anterior | Ajuste implementado |
|---|---|---|---|
| Visão Geral | Preços selecionados | O horário global e os cartões não distinguiam atualização da interface de referência de mercado. | Inclui divulgação de origem e data das séries exibidas. |
| Dashboard Global | Universo completo de commodities | Não apresentava referência e origem na tabela consolidada. | Adiciona referência e origem em cartões e tabelas, além de alerta para simulações. |
| Energia | Preços diários e proxies | Não exibia data por ativo. | O componente setorial exibe referência em cada cartão e procedência em tabela expansível. |
| Metais | Preços, proxies e séries simuladas | A simulação era identificada, mas sem referência por série. | Mantém os avisos e acrescenta data e origem. |
| Agricultura | Preços diários | Usava o componente setorial sem metadados de referência. | Recebe a mesma divulgação compartilhada de procedência. |
| Commodities Brasileiras | Preços de referência e tabela estática | A tabela qualitativa já alertava sobre defasagem; os preços não mostravam referência. | Mantém o alerta estático e herda a divulgação do componente setorial. |
| Macro & Correlações | FRED e eventual fallback simulado | Séries mensais podiam aparentar estar atrasadas. | Exibe frequência, origem e data; explica a diferença entre publicação mensal e cotação de preço. |
| Risk Analytics | Série do ativo escolhido | Aviso genérico não mostrava a data usada no cálculo. | Mostra fonte e referência da série base antes das métricas de risco. |
| Forecast | Série do ativo escolhido | O usuário não via a referência usada pelos cenários. | Mostra a procedência da base; os cenários alertam claramente quando herdam uma série simulada. |
| Quant Research | Série do ativo escolhido | O ajuste dos modelos não evidenciava a procedência da amostra. | Mostra a fonte e a última referência antes da estimação. |
| Portfolio | Painel de ativos selecionados | A otimização não informava a origem ou a referência de cada componente. | Mostra uma tabela de procedência antes da construção do painel. |

## Mudanças implementadas

A implementação introduz um componente compartilhado que calcula a última observação de cada série, traduz a origem interna para linguagem adequada à interface e expõe uma tabela de procedência. A divulgação é aplicada às páginas que usam preços e às páginas macro, o que evita repetir lógica de data e status em cada aba.[2]

| Componente | Mudança |
|---|---|
| `utils/data_status.py` | Nova camada de apresentação de fonte, frequência, referência e situação da série. |
| `utils/sector_view.py` | Cartões setoriais exibem a referência; tabelas incluem origem e data. |
| `pages/1_🌍_Dashboard_Global.py` | Tabelas e cartões consolidados passam a expor referência e origem. |
| `pages/6_🔗_Macro_Correlations.py` | Inclui frequência e data de publicação para cada indicador. |
| Risk, Forecast, Quant Research e Portfolio | Expõem a procedência da série antes de cálculos e projeções. |
| `utils/design_system.py` e `app.py` | Removem o indicador ambíguo de mercado e o relógio que poderia ser confundido com atualização de preço. |
| Documentação e testes | Removem referências de release e registram o novo contrato de transparência. |

## Limites e dependências

A transparência visual não substitui dados. Para que a aplicação publicada mostre preços de mercado, o ambiente de hospedagem precisa conseguir consultar o Yahoo Finance. Para macroeconomia, é necessário configurar `FRED_API_KEY`; além disso, a data exibida deve respeitar a frequência e o calendário de divulgação de cada série. Indicadores fundamentais, curva de futuros e os ativos atualmente simulados exigem provedores adicionais antes de serem tratados como dados de mercado.

## Validação

A suíte automatizada foi executada após as alterações com `pytest tests/ -q --disable-warnings --maxfail=1`.

| Métrica | Resultado |
|---|---:|
| Testes aprovados | **82** |
| Testes com falha | **0** |
| Tempo de execução | **14,13 s** |

## Referências

[1]: runtime_tab_audit_notes.md "Observações da aplicação publicada"
[2]: ../utils/data_status.py "Componente de transparência de dados"
[3]: freshness_audit_2026-08-12.csv "Diagnóstico de disponibilidade das séries de preço"
[4]: ../config/settings.py "Universo de ativos e frequência macro"
