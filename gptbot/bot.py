"""
ChatGPT wrapper
"""

from __future__ import annotations
import httpx
import ujson as json
import tiktoken
import math
from typing import Any, Iterable, Self, AsyncGenerator, Literal, Generator, overload
from typing import TypeAlias, cast, Sequence
from enum import Enum
from datetime import datetime
from warnings import warn
from pydantic import BaseModel, Field, validate_call, PrivateAttr
from pydantic import field_validator, model_validator, ConfigDict
from pydantic import field_serializer
from vermils.asynctools import sync_await
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from base64 import b64encode, b64decode
from io import BytesIO
from vermils.io import aio


class Model(str, Enum):
    GPT4_0125_Preview = "gpt-4-0125-preview"
    GPT4TurboPreview = "gpt-4-turbo-preview"
    GPT4_1106_Preview = "gpt-4-1106-preview"
    GPT4VisionPreview = "gpt-4-vision-preview"
    GPT4_1106_VisionPreview = "gpt-4-1106-vision-preview"
    GPT4 = "gpt-4"
    GPT4_0613 = "gpt-4-0613"
    GPT4_32K = "gpt-4-32k"
    GPT4_32K_0613 = "gpt-4-32k-0613"

    GPT3_5Turbo_0125 = "gpt-3.5-turbo-0125"
    GPT3_5Turbo = "gpt-3.5-turbo"
    GPT3_5Turbo_1106 = "gpt-3.5-turbo-1106"
    GPT3_5TurboInstruct = "gpt-3.5-turbo-instruct"

    GPT3_5Turbo_16K = "gpt-3.5-turbo-16k"
    GPT3_5Turbo_0613 = "gpt-3.5-turbo-0613"
    GPT3_5Turbo_16K_0613 = "gpt-3.5-turbo-16k-0613"


class ModelInfo(BaseModel):
    max_tokens: int
    updated_at: datetime
    legacy: bool = False


INFO_MAP: dict[Model, ModelInfo] = {
    Model.GPT4_0125_Preview: ModelInfo(max_tokens=128000, updated_at=datetime(2023, 4, 1)),
    Model.GPT4TurboPreview: ModelInfo(max_tokens=128000, updated_at=datetime(2023, 4, 1)),
    Model.GPT4_1106_Preview: ModelInfo(max_tokens=128000, updated_at=datetime(2023, 4, 1)),
    Model.GPT4VisionPreview: ModelInfo(max_tokens=128000, updated_at=datetime(2023, 4, 1)),
    Model.GPT4_1106_VisionPreview: ModelInfo(max_tokens=128000, updated_at=datetime(2023, 4, 1)),
    Model.GPT4: ModelInfo(max_tokens=8192, updated_at=datetime(2021, 9, 1)),
    Model.GPT4_0613: ModelInfo(max_tokens=8192, updated_at=datetime(2021, 9, 1)),
    Model.GPT4_32K: ModelInfo(max_tokens=32768, updated_at=datetime(2021, 9, 1)),
    Model.GPT4_32K_0613: ModelInfo(max_tokens=32768, updated_at=datetime(2021, 9, 1)),
    Model.GPT3_5Turbo_0125: ModelInfo(max_tokens=16385, updated_at=datetime(2021, 9, 1)),
    Model.GPT3_5Turbo: ModelInfo(max_tokens=4096, updated_at=datetime(2021, 9, 1)),
    Model.GPT3_5Turbo_1106: ModelInfo(max_tokens=16385, updated_at=datetime(2021, 9, 1)),
    Model.GPT3_5TurboInstruct: ModelInfo(max_tokens=4096, updated_at=datetime(2021, 9, 1)),

    Model.GPT3_5Turbo_16K: ModelInfo(max_tokens=16385, updated_at=datetime(2021, 9, 1), legacy=True),
    Model.GPT3_5Turbo_0613: ModelInfo(max_tokens=4096, updated_at=datetime(2021, 9, 1), legacy=True),
    Model.GPT3_5Turbo_16K_0613: ModelInfo(max_tokens=16385, updated_at=datetime(2021, 9, 1), legacy=True),
}


class Role(str, Enum):
    User = "user"
    Assistant = "assistant"
    System = "system"


class Message(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="allow")

    class BaseSegment(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: str

    class TextSegment(BaseSegment):
        model_config = ConfigDict(extra="allow")
        type: Literal["text"] = "text"
        text: str

        def __str__(self):
            return self.text

    class ImageSegment(BaseSegment):
        model_config = ConfigDict(extra="allow")

        class ImageURL(BaseModel):
            model_config = ConfigDict(extra="allow")
            url: str
            detail: Literal["high", "low", "auto"] = "auto"
        type: Literal["image_url"] = "image_url"
        image_url: ImageURL
        _cache: Image.Image | None = None
        _dims: tuple[int, int] = (0, 0)

        @field_validator("image_url", mode="before")
        def convert_url(cls, value: str | ImageURL | dict) -> ImageURL:
            if isinstance(value, str):
                return cls.ImageURL(url=value)
            if not isinstance(value, cls.ImageURL):
                return cls.ImageURL.model_validate(value)
            return value

        @field_serializer("image_url", when_used="json")
        def to_b64(self, img_url: ImageURL, _) -> dict:
            url = img_url.url
            if url.startswith("base64://"):
                url = f"data:image/jpeg;base64,{self.image_url.url[9:]}"
            if self._cache is not None:
                img = self._cache
                jpeg_data = BytesIO()
                img.convert("RGB").save(jpeg_data, format="jpeg",
                                        optimize=True, quality=85)
                jpeg_data.seek(0)
                url = f"data:image/jpeg;base64,{
                    b64encode(jpeg_data.read()).decode('utf-8')}"
            return self.ImageURL(url=url, detail=img_url.detail).model_dump(mode="json")

        def __str__(self):
            return f"#image({self.image_url.url[:10]}...{self.image_url.url[-10:]})"

    role: Role
    name: str | None = None
    content: str | list[TextSegment | ImageSegment] = ""

    def __str__(self):
        if isinstance(self.content, str):
            return self.content
        return ''.join(str(seg) for seg in self.content)


SegTypes: TypeAlias = Message.TextSegment | Message.ImageSegment
Prompt: TypeAlias = str | list[SegTypes]


class FullChunkResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    class Choice(BaseModel):
        model_config = ConfigDict(extra="allow")

        class Delta(BaseModel):
            model_config = ConfigDict(extra="allow")
            content: str | None = None
            tool_calls: list | None = None
            role: Role | None = None
        finish_reason: str | None
        index: int
        delta: Delta
        logprobs: dict[str, Any] | None = None
    id: str = ""
    choices: list[Choice]
    created: int
    model: str = ""
    system_fingerprint: str | None = None
    obj: str = Field(default="", alias="object")


class FullResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    class Choice(BaseModel):
        model_config = ConfigDict(extra="allow")
        finish_reason: str | None
        index: int
        message: Message
        logprobs: dict[str, Any] | None = None

    class Usage(BaseModel):
        model_config = ConfigDict(extra="allow")
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int
    id: str = ""
    choices: list[Choice]
    created: int
    model: str = ""
    system_fingerprint: str | None = None
    obj: str = Field(default="", alias="object")
    usage: Usage


class Bot(BaseModel):
    """
    # ChatGPT bot

    ## Parameters
    - `model` (Model): Model to use
    - `api_key` (str): OpenAI API key
    - `prompt` (str): Initial prompt
    - `temperature` (float): The higher the temperature, the crazier the text (0.0 to 1.0)
    - `comp_tokens` (float | int): Reserved tokens for completion, when value is in (0, 1), it represents a ratio,
        [1,-] represents tokens count, 0 for auto mode
    - `top_p` (float): Nucleus sampling: limits the generated guesses to a cumulative probability. (0.0 to 1.0)
    - `frequency_penalty` (float): Adjusts the frequency of words in the generated text. (0.0 to 1.0)
    - `presence_penalty` (float): Adjusts the presence of words in the generated text. (0.0 to 1.0)
    - `proxies` (dict[str, str]): Connection proxies
    - `timeout` (float | None): Connection timeout
    """
    model_config = ConfigDict(validate_assignment=True)  # Pydantic model config
    model: Model
    api_key: str
    api_url: str = "https://api.openai.com/v1/chat/completions"
    comp_tokens: float | int = 0

    # See https://platform.openai.com/docs/api-reference/chat/create
    prompt: str = ""
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None
    temperature: float = 0.5
    top_p: float = 1.0
    tools: list[dict[str, Any]] | None = None
    tool_choice: Literal["auto", "none"] | dict[str, Any] = "auto"
    user: str | None = None

    proxy: str | None = None
    timeout: float | None = None
    # retries: int = 5
    # backoff: float = 1.5
    _cli: httpx.AsyncClient = PrivateAttr(default=None)
    _last_proxy: str | None = PrivateAttr(default=None)
    _last_timeout: float | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def post_init(self) -> Self:
        if (cast(None | httpx.AsyncClient, self._cli) is None
                or self.proxy != self._last_proxy or self.timeout != self._last_timeout):
            self.respawn_cli()
        return self

    @field_validator("model")
    def warn_legacy(cls, value: Model) -> Model:
        if INFO_MAP[value].legacy:
            warn(f"{value} is a legacy model", DeprecationWarning)
        return value

    @field_validator("comp_tokens")
    def check_range(cls, value: int | float):
        return max(0, value)

    def respawn_cli(self, **kw):
        """
        Create a new HTTP client, replacing the old one if it exists.
        """
        self._last_proxy = self.proxy
        self._last_timeout = self.timeout
        self._cli = httpx.AsyncClient(proxy=self.proxy,
                                      timeout=self.timeout,
                                      trust_env=False,
                                      **kw)

    def new_session(
            self, messages: Iterable[Message] = tuple()) -> Session:
        return Session(bot=self, messages=list(messages))

    async def stream_raw(self,
                         prompt: Prompt,
                         role: Role = Role.User,
                         ensure_json: bool = False,
                         choices: int = 1,
                         session: Session | None = None
                         ) -> AsyncGenerator[FullChunkResponse, None]:
        session = self.new_session() if session is None else session
        session.append(Message(
            role=role.value, content=await self._proc_prompt(prompt)))
        used_tokens = session.trim(bot=self)

        async with self._cli.stream(
            "POST",
            self.api_url,
            headers={"Authorization": "Bearer " + self.api_key},
            json=self._get_json(
                session,
                ensure_json=ensure_json, stream=True, choices=choices, used_tokens=used_tokens)
        ) as r:
            if r.status_code != 200:
                session.pop()
                await r.aread()
                raise RuntimeError(
                    f"{r.status_code} {r.reason_phrase} {r.text}",
                )

            async for line in r.aiter_lines():
                if not (line := line.strip()):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                ret = FullChunkResponse.model_validate(json.loads(data))
                yield ret

    async def stream(self,
                     prompt: Prompt,
                     role: Role = Role.User,
                     ensure_json: bool = False,
                     session: Session | None = None
                     ) -> AsyncGenerator[str, None]:
        """
        Stream messages from the bot.

        ## Parameters
        - `prompt` (str): What to say
        - `role` (Role): Role of the speaker
        - `ensure_json` (bool): Ensure the response is a valid JSON object
        - `session` (Session): Session to use
        """

        content = ''
        async for r in self.stream_raw(prompt, role, ensure_json, 1, session):
            choice = r.choices[0]
            if choice.finish_reason is not None:
                continue
            if choice.delta.role is not None:
                role = choice.delta.role
            if choice.delta.content is not None:
                content += choice.delta.content
                yield choice.delta.content
        if session is not None:
            session.append(Message(role=role.value, content=content))

    def stream_sync(self,
                    prompt: Prompt,
                    role: Role = Role.User,
                    ensure_json: bool = False,
                    ) -> Generator[str, None, None]:
        raise NotImplementedError

    async def send_raw(self,
                       prompt: Prompt,
                       role: Role = Role.User,
                       ensure_json: bool = False,
                       choices: int = 1,
                       session: Session | None = None
                       ) -> FullResponse:
        session = self.new_session() if session is None else session
        session.append(Message(
            role=role.value, content=await self._proc_prompt(prompt)))
        used_tokens = session.trim(bot=self)

        r = await self._cli.post(
            self.api_url,
            headers={"Authorization": "Bearer " + self.api_key},
            json=self._get_json(
                session,
                ensure_json=ensure_json, choices=choices, used_tokens=used_tokens)
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"{r.status_code} {r.reason_phrase} {r.text}",
            )
        ret = FullResponse.model_validate(json.loads(r.text))
        return ret

    async def send(self,
                   prompt: Prompt,
                   role: Role = Role.User,
                   ensure_json: bool = False,
                   session: Session | None = None
                   ) -> str:
        """
        Send a message to the bot.

        ## Parameters
        - `prompt` (str): What to say
        - `role` (Role): Role of the speaker
        - `ensure_json` (bool): Ensure the response is a valid JSON object
        - `session` (Session): Session to use
        """
        r = await self.send_raw(prompt, role, ensure_json, 1, session)
        content = cast(str, r.choices[0].message.content)
        if session is not None:
            session.append(Message(role=role.value, content=content))
        return content

    def send_sync(self,
                  prompt: str,
                  role: Role = Role.User,
                  ensure_json: bool = False,
                  ) -> str | list[SegTypes]:
        """
        Send a message to the bot.

        ## Parameters
        - `prompt` (str): What to say
        - `role` (Role): Role of the speaker
        - `ensure_json` (bool): Ensure the response is a valid JSON object
        """
        return sync_await(self.send(prompt, role, ensure_json))

    async def cache_image_seg(
            self,
            seg: Message.ImageSegment,
            exheaders: dict | None = None,
    ) -> Image.Image:
        """
        Download and cache the image in the segment.
        """
        if exheaders is None:
            exheaders = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            }
        if seg._cache is not None:
            return seg._cache
        img_url = seg.image_url.url

        try:
            if img_url.startswith("file://"):
                img_path = Path(img_url[7:])
                if not await aio.path.exists(img_path):
                    raise ValueError(f"File not found: {img_path}")
                async with aio.open(img_path, 'rb') as f:
                    img_data = await f.read()
                img = Image.open(BytesIO(img_data))
            elif img_url.startswith("base64://"):
                img_data = b64decode(img_url[9:])
                img = Image.open(BytesIO(img_data))
            elif img_url.startswith("http"):
                r = await self._cli.get(img_url, headers=exheaders)
                if r.status_code == 404:
                    raise ValueError(f"Image not found: {img_url}")
                r.raise_for_status()
                img = Image.open(BytesIO(r.content))
            else:
                raise ValueError(f"Invalid image URL: {img_url}")
        except UnidentifiedImageError:
            raise ValueError(f"Unknown image format: {img_url}")
        seg._cache = img
        seg._dims = img.size
        return img

    def _get_json(self,
                  session: Session,
                  ensure_json: bool = False,
                  stream: bool = False,
                  choices: int = 1,
                  used_tokens: int = 0
                  ) -> dict[str, Any]:

        total_tokens = INFO_MAP[self.model].max_tokens
        spare_tokens = total_tokens - used_tokens
        if self.comp_tokens == 0:
            comp_tokens = spare_tokens
        elif self.comp_tokens < 1:
            comp_tokens = int(min(self.comp_tokens*total_tokens, spare_tokens))
        else:
            comp_tokens = int(min(self.comp_tokens, spare_tokens))
        msgs: list[str, dict[str, Any]] = [{"role": "system", "content": self.prompt}]
        msgs.extend(m.model_dump(mode="json", exclude_none=True)
                    for m in session.messages)
        ret: dict[str, Any] = {
            "messages": msgs,
            "model": self.model.value,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "logprobs": self.logprobs,
            "max_tokens": min(4096, comp_tokens),
            # limited by openai but not specified anythere...
            "n": choices,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "stop": self.stop,
            "stream": stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "user": self.user,
        }

        def remove_null(d: dict):
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, dict):
                    remove_null(v)
                elif isinstance(v, list | tuple):
                    for i in v:
                        if isinstance(i, dict):
                            remove_null(i)

        remove_null(ret)

        if self.tools is not None:
            ret["tools"] = self.tools
            ret["tool_choice"] = self.tool_choice

        if (ensure_json):
            ret["response_format"] = {
                "type": "json_object"
            }

        return ret

    async def _proc_prompt(self, prompt: Prompt) -> str | list[SegTypes]:
        if isinstance(prompt, str):
            return prompt
        ret = list[SegTypes]()
        for seg in prompt:
            if isinstance(seg, Message.ImageSegment):
                await self.cache_image_seg(seg)
            ret.append(seg)

        return ret


class Session(Sequence, BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    bot: Bot | None = Field(exclude=True)
    messages: list[Message] = Field(default_factory=list)

    def trim(self, target_max: int | None = None, bot: Bot | None = None) -> int:
        """
        Trim the session until it's less than `target_max` tokens.
        Returns the total number of tokens after trimming.
        """
        bot = self.bot if bot is None else bot
        if bot is None:
            raise ValueError("bot is not set")
        if target_max is None:
            model_max_tokens = INFO_MAP[bot.model].max_tokens
            if bot.comp_tokens < 1:
                target_max = int((1-bot.comp_tokens) * model_max_tokens)
            else:
                target_max = max(0, model_max_tokens-int(bot.comp_tokens))

        # modified from official doc: https://platform.openai.com/docs/guides/text-generation/managing-tokens
        try:
            encoding = tiktoken.encoding_for_model(bot.model.value)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        def sizeof_prompt(prompt: str | list[SegTypes]) -> int:
            if isinstance(prompt, str):
                return len(encoding.encode(prompt))
            size = 0
            for seg in prompt:
                if isinstance(seg, Message.TextSegment):
                    size += len(encoding.encode(seg.text))
                elif isinstance(seg, Message.ImageSegment):
                    # from https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision/492318
                    size += 85
                    if seg.image_url.detail != "low":
                        size += 170 * \
                            math.ceil(seg._dims[0] / 512) * \
                            math.ceil(seg._dims[1] / 512)
                else:
                    raise ValueError(f"Invalid segment type: {seg['type']}")
            return size
        num_tokens = 4 + sizeof_prompt(bot.prompt)
        num_tokens += 2  # every reply is primed with <im_start>assistant
        if num_tokens >= target_max:
            raise ValueError("The prompt is already too long")
        msg_cnt = 0
        for message in reversed(self.messages):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            this_tokens = 4
            if message.name is not None:
                this_tokens -= 1
            this_tokens += sizeof_prompt(message.role)
            this_tokens += sizeof_prompt(message.content)
            if num_tokens + this_tokens >= target_max:
                break
            msg_cnt += 1
            num_tokens += this_tokens
        self.messages = self.messages[len(self.messages)-msg_cnt:]
        return num_tokens

    @validate_call
    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def pop(self, index: int = -1) -> Message:
        return self.messages.pop(index)

    def rollback(self, num: int):
        """
        Roll back `num` messages.
        """
        self.messages = self.messages[:len(self.messages)-num]

    def clear(self):
        """
        Clear the session.
        """
        self.messages.clear()

    @validate_call
    async def stream(self,
                     prompt: Prompt,
                     role: Role = Role.User,
                     ensure_json: bool = False,
                     ) -> AsyncGenerator[str, None]:
        """
        Stream messages from the bot.

        ## Parameters
        - `prompt` (str): What to say
        - `role` (Role): Role of the speaker
        - `ensure_json` (bool): Ensure the response is a valid JSON object
        - `session` (Session): Session to use
        """
        if self.bot is None:
            raise ValueError("bot is not set")
        async for r in self.bot.stream(prompt, role, ensure_json, self):
            yield r

    @validate_call
    async def send(self,
                   prompt: Prompt,
                   role: Role = Role.User,
                   ensure_json: bool = False,
                   ) -> str:
        """
        Send a message to the bot.

        ## Parameters
        - `prompt` (str): What to say
        - `role` (Role): Role of the speaker
        - `ensure_json` (bool): Ensure the response is a valid JSON object
        """
        if self.bot is None:
            raise ValueError("bot is not set")
        return await self.bot.send(prompt, role, ensure_json, self)

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def __bool__(self):
        return bool(self.messages)

    def __reversed__(self):
        return reversed(self.messages)

    @overload
    def __getitem__(self, index: int) -> Message:
        ...

    @overload
    def __getitem__(self, s: slice) -> list[Message]:
        ...

    def __getitem__(self, index: int | slice):
        return self.messages[index]

    @validate_call
    def __setitem__(self, index: int, value: Message):
        self.messages[index] = value
