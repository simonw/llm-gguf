import click
import httpx
import json
from llama_cpp import Llama
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


@llm.hookimpl
def register_models(register):
    models_file = _ensure_models_file()
    models = json.loads(models_file.read_text())
    for model_id, info in models.items():
        model_path = info["path"]
        aliases = info.get("aliases", [])
        model_id = f"gguf/{model_id}"
        model = GgufChatModel(model_id, model_path)
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
        with httpx.stream("GET", url, follow_redirects=True) as response:
            total_size = response.headers.get("content-length")

            filename = url.split("/")[-1]
            download_path = _ensure_models_dir() / filename
            if download_path.exists():
                raise click.ClickException(f"File already exists at {download_path}")

            with open(download_path, "wb") as fp:
                if total_size is not None:  # If Content-Length header is present
                    total_size = int(total_size)
                    with click.progressbar(
                        length=total_size,
                        label="Downloading {}".format(human_size(total_size)),
                    ) as bar:
                        for data in response.iter_bytes(1024):
                            fp.write(data)
                            bar.update(len(data))
                else:  # If Content-Length header is not present
                    for data in response.iter_bytes(1024):
                        fp.write(data)

            click.echo(f"Downloaded model to {download_path}", err=True)
            models_file = _ensure_models_file()
            models = json.loads(models_file.read_text())
            model_id = download_path.stem
            info = {
                "path": str(download_path.resolve()),
                "aliases": aliases,
            }
            models[model_id] = info
            models_file.write_text(json.dumps(models, indent=2))

    @gguf.command()
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
    def register_model(filepath, aliases):
        "Register a GGUF model that you have already downloaded with LLM"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        path = pathlib.Path(filepath)
        model_id = path.stem
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
            info["size"] = human_size(pathlib.Path(info["path"]).stat().st_size)
        click.echo(json.dumps(models, indent=2))


class GgufChatModel(llm.Model):
    can_stream = True

    def __init__(self, model_id, model_path):
        self.model_id = model_id
        self.model_path = model_path
        self._model = None

    def get_model(self):
        if self._model is None:
            self._model = Llama(
                model_path=self.model_path, verbose=False, n_ctx=0  # "0 = from model"
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
            completion = model.create_chat_completion(messages=messages)
            return [completion["choices"][0]["text"]]

        # Streaming
        model = self.get_model()
        completion = model.create_chat_completion(messages=messages, stream=True)
        for chunk in completion:
            choice = chunk["choices"][0]
            delta_content = choice.get("delta", {}).get("content")
            if delta_content is not None:
                yield delta_content


def human_size(num_bytes):
    """Return a human readable byte size."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num_bytes < 1024.0:
            break
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} {unit}"
