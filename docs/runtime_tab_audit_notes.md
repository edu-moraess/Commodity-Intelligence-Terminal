# Notas de auditoria das abas em produção

## 12 de agosto de 2026 — observação inicial

A URL publicada `https://commodity-intelligence-terminal.streamlit.app/` foi acessada no navegador. Na primeira e na segunda tentativa, a interface ainda apresentava apenas a tela de carregamento do Streamlit, sem conteúdo de aplicação ou navegação entre abas disponível para inspeção.

A próxima etapa da auditoria será feita em paralelo à revisão estática das páginas e da camada de dados do repositório, distinguindo a data observada da origem de cada série e a indicação de fonte sintética.

## Evidências visuais — Visão Geral e Dashboard Global

Após o carregamento, a barra lateral da aplicação exibiu um carimbo de atualização `12/08/2026 12:21 BRT` e um indicador geral de disponibilidade de mercado. A tela inicial mostrou um gráfico de retorno acumulado que alcança agosto de 2026, mas não exibe no próprio gráfico a última observação por ativo nem a fonte de cada série.

Na aba **Dashboard Global**, a própria interface informa que **22/22 ativos** estão sendo exibidos como **dados simulados**, pois o provedor de preços não respondeu naquele ambiente. Portanto, o indicador geral e o carimbo de atualização não representam, por si só, a disponibilidade de cotações externas reais. Esta é uma comunicação ambígua e deve ser corrigida.

## Evidências visuais — Energy Analytics

A aba **Energy Analytics** apresenta cartões e métricas de Brent, WTI, gás natural, gasolina, diesel, carvão e urânio, mas não mostra a data da última observação, o horário de coleta, a frequência de atualização nem a proveniência ao lado dos valores. O rodapé lateral continuou a mostrar o mesmo carimbo global de `12/08/2026 12:22 BRT`, que pode ser apenas o horário da aplicação/cache e não a data de mercado de cada ativo.

A aba documenta que módulos de produção OPEP/EUA, estoques EIA, rig count e curva de futuros ainda requerem integrações externas. O clique inicial para outra aba permaneceu na rota de Energy; as próximas abas serão acessadas por rotas diretas, evitando depender desse comportamento de navegação durante o carregamento.

## Evidências visuais — Metals Analytics

A aba **Metals Analytics** declara que ouro, prata e cobre têm referência no Yahoo Finance, enquanto alumínio e minério de ferro são apresentados como dados sintéticos e lítio, níquel e zinco como proxies. A interface mostra um aviso de que uma ou mais séries são simuladas porque o provedor não respondeu no ambiente, além de identificadores `simulado` em cartões específicos.

Apesar da transparência sobre simulação e proxy, os cartões e a tabela de métricas não exibem a data da última observação de cada série nem distinguem o horário de atualização do aplicativo do timestamp de mercado. O indicador lateral geral permanece ambíguo nesse cenário.

## Tentativa de inspeção — Agriculture Analytics

A navegação direta para `/Agriculture` falhou durante o carregamento e retornou uma página em branco. Por isso, a aba de agricultura será auditada pelo código-fonte e por uma nova tentativa de navegação a partir da aplicação raiz. A falha de roteamento/carregamento é, por si, um ponto que exige validação após o próximo deploy.
