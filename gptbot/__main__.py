import argparse
import asyncio
import ujson as json
import urllib.request
import re
from traceback import format_exc
from pathlib import Path
from getpass import getpass
from gptbot import bot, __author__, __version__
from platformdirs import user_cache_dir, user_config_dir


CONFIG_VERSION = "0.2.0"
SESSION_VERSION = "0.3.0"


async def main():
    parser = argparse.ArgumentParser(description="GPTBot")

    cache_dir = Path(user_cache_dir(
        "gptbot", __author__, SESSION_VERSION, ensure_exists=True))
    config_dir = Path(user_config_dir(
        "gptbot", __author__, CONFIG_VERSION, ensure_exists=True))

    parser.add_argument("-m", "--model", help="Model name",
                        choices=list(e.value for e in bot.Model), default=None)
    parser.add_argument("-c", "--config", help="Path to config file",
                        default=config_dir / "config.json")
    parser.add_argument("-s", "--session",
                        help="Path to session file", default=cache_dir / "session.json")
    parser.add_argument("-k", "--create-config",
                        help="Create a new config file if not existed", action="store_true")
    parser.add_argument("-V", "--version", action="version",
                        version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    model = args.model
    config_path = Path(args.config)
    session_path = Path(args.session)
    create_config = args.create_config

    if not config_path.exists():
        if create_config:
            print(f"Creating config file: {config_path}")
        else:
            print(f"Config file not found: {config_path}")

        if not model:
            print("Choose a model from the following list:")
            models = tuple(bot.Model)
            for i, m in enumerate(models):
                print(f"    {i:2}: {m.value}")
            try:
                model = models[int(input("Enter the number of the model: "))]
                print(f"Selected model: {model.value}")
            except (ValueError, IndexError):
                print("Invalid input, exiting")
                exit(1)

        api_key = getpass("Enter your OpenAI API key: ")
        b = bot.Bot(model=model, api_key=api_key, timeout=20)

        if create_config:
            config_path.write_text(b.model_dump_json(indent=4))
            print(f"Config file created: {config_path}")
            return
    else:
        if create_config:
            print(f"Config file already exists: {config_path}")
            return
        print(f"Loading config file: {config_path}")
        b = bot.Bot.model_validate_json(config_path.read_text())

        if model:
            b.model = model

    if not session_path.exists():
        print(f"Session file not found: {session_path}")
        print("Starting new session")
        session = b.new_session()
    else:
        print(f"Loading session file: {session_path}")
        with open(session_path, 'r') as f:
            session = bot.Session(bot=b, **json.load(f))

    if not b.api_key:
        b.api_key = getpass("Enter your OpenAI API key: ")

    if b.proxy is None:
        sys_proxies = urllib.request.getproxies()
        proxy = sys_proxies.get("https", sys_proxies.get("http", None))
        b.proxy = proxy
        print(f"Using system proxy: {b.proxy}")

    print("\nChat started:", end="\n\n")
    print(f"System: {b.prompt}", end="\n\n")
    namemap = {
        "user": "You",
        "assistant": "Bot",
        "system": "System",
    }
    revisit_n = 2
    if len(session) > revisit_n + 1:
        print("...", end="\n\n")
    for h in session[1:][-revisit_n:]:
        print(f"{namemap[h.role.value]}: {str(h)}", end="\n\n")

    try:
        while True:
            try:
                image_re = re.compile(r"(?<!#)(?:##)*#{1}image\((.*?)\)")
                role = bot.Role.User
                user_input = input("You: ")
                print('')

                if user_input.startswith('/'):
                    match user_input[1:].split():
                        case ["exit"]:
                            break
                        case ["save"]:
                            session_path.write_text(session.model_dump_json(indent=4))
                            print(f"Session saved to {session_path}")
                        case ["save", path]:
                            session_path = Path(path)
                            session_path.write_text(session.model_dump_json(indent=4))
                            print(f"Session saved to {session_path}")
                        case ["load"]:
                            with open(session_path, 'r') as f:
                                session = bot.Session(bot=b, **json.load(f))
                            print(f"Session loaded from {session_path}")
                        case ["load", path]:
                            session_path = Path(path)
                            with open(session_path, 'r') as f:
                                session = bot.Session(bot=b, **json.load(f))
                            print(f"Session loaded from {session_path}")
                        case ["clear"]:
                            session.clear()
                            print("Session cleared")
                        case ["rollback"]:
                            session.rollback(1)
                            print("Session rolled back")
                        case ["rollback", num]:
                            session.rollback(int(num))
                            print(f"Session rolled back {num} steps")
                        case ["role"]:
                            print(f"Current role {role.value}")
                            print("Choose a role from the following list:")
                            roles = tuple(bot.Role)
                            for i, r in enumerate(roles):
                                print(f"    {i:2}: {r.value}")
                            try:
                                role = roles[int(
                                    input("Enter the number of the role: "))]
                                print(f"Selected role: {role.value}")
                            except (ValueError, IndexError):
                                print("Invalid input")
                        case ["model"]:
                            print(f"Current model {b.model.value}")
                            print("Choose a model from the following list:")
                            models = tuple(bot.Model)
                            for i, m in enumerate(models):
                                print(f"    {i:2}: {m.value}")
                            try:
                                model = models[int(
                                    input("Enter the number of the model: "))]
                                print(f"Selected model: {model.value}")
                                b.model = model
                            except (ValueError, IndexError):
                                print("Invalid input")
                        case ["tokens"]:
                            print(f"Used tokens(approximately): {session.trim()}")
                        case ["help"]:
                            print("Commands:")
                            print("\t/exit")
                            print("\t/save")
                            print("\t/save <path>")
                            print("\t/load")
                            print("\t/load <path>")
                            print("\t/clear")
                            print("\t/rollback")
                            print("\t/rollback <num>")
                            print("\t/role")
                            print("\t/model")
                            print("\t/tokens")
                            print("\t/help")
                            print("\tTo embed an image, use #image(<url>)")
                        case _:
                            print("Invalid command, use /help for a list of commands.")
                    print('')
                    continue

                prompt: bot.Prompt
                if match := image_re.search(user_input):
                    prompt = list[bot.SegTypes]()
                    try:
                        for img_url in match.groups():
                            text_end = user_input.index(img_url)
                            pre_text = user_input[:text_end-7]  # remove `#image(`
                            if pre_text:
                                prompt.append(bot.Message.TextSegment(text=pre_text))
                            user_input = user_input[text_end +
                                                    len(img_url) + 1:]  # remove `)`
                            img_seg = bot.Message.ImageSegment(image_url=img_url)
                            prompt.append(img_seg)
                            if user_input:
                                prompt.append(bot.Message.TextSegment(text=user_input))
                    except ValueError as e:
                        print("Invalid image URL")
                        print(f"Error: {e}")
                        print(format_exc(), end="\n\n")
                        continue
                else:
                    prompt = user_input

                print("Bot: ", end='', flush=True)
                try:
                    async for res in session.stream(prompt, role):
                        print(res, end='', flush=True)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("\n<Interrupted>")
                finally:
                    print('\n')
            except KeyboardInterrupt:
                print("Exiting")
                break
            except Exception as e:
                print(f"Error: {e}")
                print(format_exc(), end="\n\n")
                continue
    finally:
        user_input = input("Save session? (Y/n): ")
        user_input = user_input.strip() or 'y'
        if user_input.lower() == 'y':
            session_path.write_text(session.model_dump_json(indent=4))
            print(f"Session saved to {session_path}")

asyncio.run(main())
