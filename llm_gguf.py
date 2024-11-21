import click
import httpx
import json
from llama_cpp import Llama
from llama_cpp import llama_chat_format
import llm
import pathlib


def _ensure_models_dir():
    directory = llm.user_dir() / "gguf" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _ensure_models_file():
    directory = llm.user_dir() / "gguf"
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "models.json"
    if not filepath.exists():
        filepath.write_text("{}")
    return filepath


def _ensure_embed_models_file():
    directory = llm.user_dir() / "gguf"
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "embed-models.json"
    if not filepath.exists():
        filepath.write_text("{}")
    return filepath


@llm.hookimpl
def register_models(register):
    models_file = _ensure_models_file()
    models = json.loads(models_file.read_text())
    for model_id, info in models.items():
        model_path = info["path"]
        aliases = info.get("aliases", [])
        clip_model_path = info.get("clip_model_path")
        chat_handler_class = info.get("chat_handler_class")
        model_id = f"gguf/{model_id}"
        model = GgufChatModel(
            model_id,
            model_path,
            clip_model_path=clip_model_path,
            chat_handler_class=chat_handler_class,
            n_ctx=info.get("n_ctx", 0),
        )
        register(model, aliases=aliases)


@llm.hookimpl
def register_embedding_models(register):
    models_file = _ensure_embed_models_file()
    models = json.loads(models_file.read_text())
    for model_id, info in models.items():
        model_path = info["path"]
        aliases = info.get("aliases", [])
        model_id = f"gguf/{model_id}"
        model = GgufEmbeddingModel(model_id, model_path)
        register(model, aliases=aliases)


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def gguf():
        "Commands for working with GGUF models"

    @gguf.command()
    def models_file():
        "Display the path to the gguf/models.json file"
        directory = llm.user_dir() / "gguf"
        directory.mkdir(parents=True, exist_ok=True)
        models_file = directory / "models.json"
        click.echo(models_file)

    @gguf.command()
    def embed_models_file():
        "Display the path to the gguf/embed-models.json file"
        directory = llm.user_dir() / "gguf"
        directory.mkdir(parents=True, exist_ok=True)
        models_file = directory / "embed-models.json"
        click.echo(models_file)

    @gguf.command()
    def models_dir():
        "Display the path to the directory holding downloaded GGUF models"
        click.echo(_ensure_models_dir())

    @gguf.command()
    @click.argument("url")
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def download_model(url, aliases):
        "Download and register a GGUF model from a URL"
        download_gguf_model(url, _ensure_models_file, aliases)

    @gguf.command()
    @click.argument("url")
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def download_embed_model(url, aliases):
        "Download and register a GGUF embedding model from a URL"
        download_gguf_model(url, _ensure_embed_models_file, aliases)

    @gguf.command()
    @click.argument("model_id")
    @click.argument(
        "filepath", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
    )
    @click.option(
        "clip_model_path",
        "--clip-model-path",
        type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    )
    @click.option("chat_handler_class", "--chat-handler-class", type=str)
    @click.option("n_ctx", "--n-ctx", type=int, default=0)
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def register_model(
        model_id, filepath, clip_model_path, chat_handler_class, n_ctx, aliases
    ):
        "Register a GGUF model that you have already downloaded with LLM"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        path = pathlib.Path(filepath)
        info = {
            "path": str(path.resolve()),
            "aliases": aliases,
        }
        if clip_model_path:
            info["clip_model_path"] = clip_model_path
        if chat_handler_class:
            if not hasattr(llama_chat_format, chat_handler_class):
                raise click.ClickException(
                    f"Invalid chat handler class: {chat_handler_class}"
                )
            info["chat_handler_class"] = chat_handler_class
        if n_ctx:
            info["n_ctx"] = n_ctx
        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))

    @gguf.command()
    @click.argument("model_id")
    @click.argument(
        "filepath", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
    )
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def register_embed_model(model_id, filepath, aliases):
        "Register a GGUF embedding model that you have already downloaded"
        models_file = _ensure_embed_models_file()
        models = json.loads(models_file.read_text())
        path = pathlib.Path(filepath)
        info = {
            "path": str(path.resolve()),
            "aliases": aliases,
        }
        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))

    @gguf.command()
    def models():
        "List registered GGUF models"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        for model, info in models.items():
            try:
                info["size"] = human_size(pathlib.Path(info["path"]).stat().st_size)
            except FileNotFoundError:
                info["size"] = None
        click.echo(json.dumps(models, indent=2))

    @gguf.command()
    def embed_models():
        "List registered GGUF embedding models"
        models_file = _ensure_embed_models_file()
        models = json.loads(models_file.read_text())
        for model, info in models.items():
            try:
                info["size"] = human_size(pathlib.Path(info["path"]).stat().st_size)
            except FileNotFoundError:
                info["size"] = None
        click.echo(json.dumps(models, indent=2))


class GgufChatModel(llm.Model):
    can_stream = True

    def __init__(
        self,
        model_id,
        model_path,
        n_ctx=0,
        clip_model_path=None,
        chat_handler_class=None,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.chat_handler_class = chat_handler_class
        self.n_ctx = n_ctx  # "0 = from model"
        self._model = None

    def get_model(self):
        if self._model is None:
            if self.chat_handler_class is None:
                self._model = Llama(
                    model_path=self.model_path,
                    verbose=False,
                    n_ctx=self.n_ctx,
                    chat_format="chatml-function-calling",
                )
            else:
                chat_handler_class = getattr(llama_chat_format, self.chat_handler_class)
                self._model = Llama(
                    model_path=self.model_path,
                    verbose=False,
                    n_ctx=self.n_ctx,
                    chat_handler_class=chat_handler_class(
                        clip_model_path=self.clip_model_path
                    ),
                )
        return self._model

    def execute(self, prompt, stream, response, conversation):
        messages = []
        current_system = None
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})

        if not stream:
            model = self.get_model()
            completion = model.create_chat_completion(
                messages=messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search_blog",
                            "description": "Search for posts on the blog.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query, keywords only",
                                    }
                                },
                                "required": ["query"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                tool_choice="auto",
            )
            breakpoint()
            return [completion["choices"][0]["text"]]

        # Streaming
        model = self.get_model()
        completion = model.create_chat_completion(messages=messages, stream=True)
        for chunk in completion:
            choice = chunk["choices"][0]
            delta_content = choice.get("delta", {}).get("content")
            if delta_content is not None:
                yield delta_content


class GgufEmbeddingModel(llm.EmbeddingModel):
    def __init__(self, model_id, model_path):
        self.model_id = model_id
        self.model_path = model_path
        self._model = None

    def embed_batch(self, texts):
        if self._model is None:
            self._model = Llama(
                model_path=self.model_path, embedding=True, verbose=False
            )
        results = self._model.create_embedding(list(texts))
        return [result["embedding"] for result in results["data"]]


def download_gguf_model(url, models_file_func, aliases):
    """Download a GGUF model and register it in the specified models file"""
    with httpx.stream("GET", url, follow_redirects=True) as response:
        total_size = response.headers.get("content-length")

        filename = url.split("/")[-1]
        download_path = _ensure_models_dir() / filename
        if download_path.exists():
            raise click.ClickException(f"File already exists at {download_path}")

        with open(download_path, "wb") as fp:
            if total_size is not None:
                total_size = int(total_size)
                with click.progressbar(
                    length=total_size,
                    label="Downloading {}".format(human_size(total_size)),
                ) as bar:
                    for data in response.iter_bytes(1024):
                        fp.write(data)
                        bar.update(len(data))
            else:
                for data in response.iter_bytes(1024):
                    fp.write(data)

        click.echo(f"Downloaded model to {download_path}", err=True)
        models_file = models_file_func()
        models = json.loads(models_file.read_text())
        model_id = download_path.stem
        info = {
            "path": str(download_path.resolve()),
            "aliases": aliases,
        }
        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))


def human_size(num_bytes):
    """Return a human readable byte size."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num_bytes < 1024.0:
            break
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} {unit}"
