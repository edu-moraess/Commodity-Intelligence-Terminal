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

## Verificação posterior — referências ainda visíveis na implantação publicada

Na nova inspeção da URL publicada, a interface ainda exibiu `v5.0.0` no subtítulo da página inicial, no cartão informativo, no rodapé da barra lateral e no relógio global. Também continuou a apresentar o rótulo `MARKETS ONLINE` e o botão `Atualizar dados` antigos. Esses elementos correspondem ao código anterior à limpeza já registrada no repositório local; o commit local não foi enviado ao remoto e, portanto, a implantação publicada ainda não recebeu essas alterações.

## Verificação após o envio ao remoto

A implantação passou a exibir o novo subtítulo, o novo botão `Atualizar séries`, a divulgação de fonte e data e a nova barra lateral. Contudo, durante a recarga inicial, o bloco “Sobre a Plataforma” ainda apresentou textos antigos, incluindo uma referência de versão. A atualização seguinte retornou à tela de carregamento, indicando que a hospedagem ainda estava reiniciando o processo. Será necessária nova verificação quando a implantação estabilizar.

## Confirmação de implantação estabilizada

Após a reinicialização, a página inicial publicada passou a exibir integralmente os novos textos: não há identificação de versão no subtítulo, no cartão informativo, na barra lateral ou no botão de atualização. A nova divulgação de fonte e data também está visível. A rota de Forecast iniciou nova recarga e será confirmada após o carregamento completo; o código publicado dessa aba já não contém referência de versão ou fingerprint visível.
