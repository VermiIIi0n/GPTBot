# Simple Wrapper For The ChatGPT API

This module is designed for asynchronous usage and provides a simple interface to the OpenAI ChatGPT API.

It also provides a command-line interface for interactive usage.

While it does provide synchronous methods, it is recommended to use the asynchronous methods for better performance.

## Installation

```bash
pip install -U ngptbot
```

## Usage

### Command Line

#### Configuration

`gptbot` requires a model and an API key to be set.

You can use config file, command parameters or interactive input to set these values.

By default, `gptbot` will look for files named `config.json` and `session.json` in the user config and cache directory.

```bash
gptbot -c /path/to/config.json  # Use a config file
gptbot -m "gpt-3.5-turbo"  # Set model, input API key interactively
gptbot -k /path/to/config.json  # create config file
```

#### Parameters

- `-c`, `--config`: Path to a JSON file
- `-m`, `--model`: Model name
- `-k`, `--create-config`: Create a config file interactively
- `-s`, `--session`: Session history file
- `-V`, `--version`: Show version

**_Precedence_**: Interactive input > Command parameters > Config file

#### Interactive Mode

This mode mimics a chat interface. You can type your message and get a response.

Commands are started with a `/` and are **case-sensitive**.

- `/exit`: Exit the program
- `/save <path>`: Save the session history to a file
- `/load <path>`: Load a session history from a file
- `/rollback <step>`: Rollback to the previous state
- `/clear`: Clear the session history
- `/role`: Switch role
- `/model`: Switch model
- `/tokens`: Show used tokens
- `/help`: Show help

To embed an image, insert `#image(<url>)` in your message.
Use double `#` to escape embedding.

```Python
await bot.send("""
What are these?
#image(https://example.com/image.png)
#image(file:///path/to/image.png)
#image(base64://<base64-encoded-image>)
""")
```

All URLs will be downloaded and embedded as base64-encoded images.

### API

```python
import asyncio
import gptbot

bot = gptbot.Bot(model="gpt-3.5-turbo", api_key="your-api-key")

async def main():
    response = await bot.send("Hello, how are you?")
    print(response)

    async for r in bot.stream("I'm fine, thank you."):
        print(r, end='')

    async for res in bot.stream_raw("What is the answer to life, the universe, and everything?"):
        """
        stream_raw returns a stream of `FullChunkResponse`.
        It contains all raw data from the API.
        """

    bot.send_sync("Goodbye!")  # Synchronous version of `send`

    # Bot class is stateless, to keep track of the conversation, use the `Session` class
    session = bot.new_session()

    # You can use session just like the bot
    async for r in session.stream("Hello!"):
        print(r, end='')

    print(await session.send("Goodbye!"))

asyncio.run(main())
```
