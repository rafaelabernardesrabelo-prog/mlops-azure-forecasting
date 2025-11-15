# Executando o Projeto com Docker

Este projeto utiliza **Docker Compose** para gerenciar a construção (build) das imagens e a execução dos serviços. Abaixo estão os comandos necessários e suas explicações.

---

## 1. Build das imagens

Antes de iniciar a aplicação, você precisa construir as imagens Docker definidas no arquivo `docker-compose.build.yml`.

Use o comando:

```bash
docker compose -f infra/composes/docker-compose.build.yml build
```

### 🔍 O que esse comando faz?

* Utiliza o arquivo de configuração específico **docker-compose.build.yml**.
* Realiza o **build** das imagens necessárias para o ambiente.
* Garante que todas as dependências e configurações estejam prontas antes de rodar o projeto.

---

## ▶️ 2. Subir a aplicação

Após o build, basta iniciar os serviços usando o arquivo principal `docker-compose.yml`.

Execute:

```bash
docker compose -f infra/composes/docker-compose.yml up
```

### 🔍 O que esse comando faz?

* Sobe todos os containers definidos no arquivo **docker-compose.yml**.
* Inicia automaticamente os serviços do projeto.
* Exibe os logs em tempo real diretamente no terminal.

---

## Dica

Se quiser rodar tudo em modo **detached** (em segundo plano), use:

```bash
docker compose -f infra/composes/docker-compose.yml up -d
```

Assim seu terminal continuará livre enquanto os serviços rodam.
