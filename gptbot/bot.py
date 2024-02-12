"""
ChatGPT wrapper
"""

from __future__ import annotations
import httpx
import ujson as json
import tiktoken
import math
from typing import Any, Self, AsyncGenerator, Literal, Generator
from typing import TypeAlias, cast
from enum import Enum
from datetime import datetime
from uuid import uuid4
from warnings import warn
from pydantic import BaseModel, Field, validate_call, PrivateAttr
from pydantic import field_validator, model_validator, ConfigDict
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

    class ImageSegment(BaseSegment):
        model_config = ConfigDict(extra="allow")

        class ImageURL(BaseModel):
            model_config = ConfigDict(extra="allow")
            url: str
            detail: Literal["high", "low", "auto"] = "auto"
        type: Literal["image_url"] = "image_url"
        image_url: ImageURL
        _cache: Image.Image | None = None

        @field_validator("image_url", mode="before")
        def convert_url(cls, value: str | ImageURL | dict) -> ImageURL:
            if isinstance(value, str):
                return cls.ImageURL(url=value)
            if not isinstance(value, cls.ImageURL):
                return cls.ImageURL.model_validate(value)
            return value

    role: Role
    content: str | list[TextSegment | ImageSegment] = ""


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
    - `trim_thre` (float | int): When the value is above 1, it's the maximum used tokens,
        when it's below 1, it's the maximum used ratio of tokens. Session will be trimmed if condition is not met.
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
    trim_thre: float | int = 0.9

    # See https://platform.openai.com/docs/api-reference/chat/create
    prompt: str = ""
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_reply_tokens: int | None = None
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
    _sess: list[dict] = PrivateAttr(default_factory=list)
    _cli: httpx.AsyncClient = PrivateAttr(default=None)

    @property
    def session(self) -> tuple[Message, ...]:
        return (Message(role=Role.System, content=self.prompt),
                *(Message.model_validate(m) for m in self._sess))

    @session.setter
    def session(self, value: list[Message | dict]):
        self._sess = [m.model_dump() if isinstance(m, BaseModel) else m for m in value]

    @model_validator(mode="after")
    def post_init(self) -> Self:
        self.respawn_cli()
        return self

    @field_validator("model")
    def warn_legacy(cls, value: Model) -> Model:
        if INFO_MAP[value].legacy:
            warn(f"{value} is a legacy model", DeprecationWarning)
        return value

    def respawn_cli(self, **kw):
        """
        Create a new HTTP client, replacing the old one if it exists.
        """
        self._cli = httpx.AsyncClient(proxy=self.proxy,
                                      timeout=self.timeout,
                                      trust_env=False,
                                      **kw)

    def trim(self, target_max: int | None = None) -> int:
        """
        Trim the session until it's less than `target_max` tokens.
        Returns the total number of tokens after trimming.
        """
        if target_max is None:
            model_max_tokens = INFO_MAP[self.model].max_tokens
            if self.trim_thre < 1:
                target_max = int(self.trim_thre * model_max_tokens)
            else:
                target_max = min(model_max_tokens, int(self.trim_thre))

        # modified from official doc: https://platform.openai.com/docs/guides/text-generation/managing-tokens
        try:
            encoding = tiktoken.encoding_for_model(self.model.value)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        def sizeof_prompt(prompt: str | list[dict]) -> int:
            if isinstance(prompt, str):
                return len(encoding.encode(prompt))
            size = 0
            for seg in prompt:
                if seg["type"] == "text":
                    size += len(encoding.encode(seg["text"]))
                elif seg["type"] == "image_url":
                    # from https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision/492318
                    size += 85
                    if seg["image_url"]["detail"] != "low":
                        size += 170 * \
                            math.ceil(seg["dims"][0] / 512) * \
                            math.ceil(seg["dims"][1] / 512)
                else:
                    raise ValueError(f"Invalid segment type: {seg['type']}")
            return size
        num_tokens = 4 + sizeof_prompt(self.prompt)
        num_tokens += 2  # every reply is primed with <im_start>assistant
        if num_tokens >= target_max:
            raise ValueError("The prompt is already too long")
        msg_cnt = 0
        for message in reversed(self._sess):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            this_tokens = 4
            for key, value in message.items():
                this_tokens += sizeof_prompt(value)
                if key == "name":  # if there's a name, the role is omitted
                    this_tokens -= 1  # role is always required and always 1 token
            if num_tokens + this_tokens >= target_max:
                break
            msg_cnt += 1
            num_tokens += this_tokens
        self._sess = self._sess[len(self._sess)-msg_cnt:]
        return num_tokens

    def rollback(self, num: int):
        """
        Roll back `num` messages.
        """
        self._sess = self._sess[:len(self._sess)-num]

    def clear(self):
        """
        Clear the session.
        """
        self._sess.clear()

    @validate_call
    async def stream_raw(self,
                         prompt: Prompt,
                         role: Role = Role.User,
                         ensure_json: bool = False,
                         choices: int = 1
                         ) -> AsyncGenerator[FullChunkResponse, None]:
        self._sess.append({
            "role": role.value, "content": await self._proc_prompt(prompt)})
        n = self.trim()
        print(f"Trimmed to {n} tokens")

        async with self._cli.stream(
            "POST",
            self.api_url,
            headers={"Authorization": "Bearer " + self.api_key},
            json=self._get_json(ensure_json, stream=True, choices=choices)
        ) as r:
            if r.status_code != 200:
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
        """

        msg = {"role": role.value, "content": ""}
        async for r in self.stream_raw(prompt, role, ensure_json, 1):
            choice = r.choices[0]
            if choice.finish_reason is not None:
                continue
            if choice.delta.role is not None:
                msg["role"] = choice.delta.role.value
            if choice.delta.content is not None:
                msg["content"] += choice.delta.content
                yield choice.delta.content
        self._sess.append(msg)

    @validate_call
    def stream_sync(self,
                    prompt: Prompt,
                    role: Role = Role.User,
                    ensure_json: bool = False,
                    ) -> Generator[str, None, None]:
        raise NotImplementedError

    @validate_call
    async def send_raw(self,
                       prompt: Prompt,
                       role: Role = Role.User,
                       ensure_json: bool = False,
                       choices: int = 1
                       ) -> FullResponse:
        self._sess.append({
            "role": role.value, "content": await self._proc_prompt(prompt)})
        self.trim()

        r = await self._cli.post(
            self.api_url,
            headers={"Authorization": "Bearer " + self.api_key},
            json=self._get_json(ensure_json, choices=choices)
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"{r.status_code} {r.reason_phrase} {r.text}",
            )
        ret = FullResponse.model_validate(json.loads(r.text))
        return ret

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
        r = await self.send_raw(prompt, role, ensure_json, 1)
        return cast(str, r.choices[0].message.content)

    @validate_call
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
        return img

    async def get_b64url_from_seg(
        self,
        seg: Message.ImageSegment,
    ):
        if seg.image_url.url.startswith("base64://"):
            return f"data:image/jpeg;base64,{seg.image_url.url[9:]}"
        # return seg.image_url.url
        img = await self.cache_image_seg(seg)
        jpeg_data = BytesIO()
        img.convert("RGB").save(jpeg_data, format="jpeg", optimize=True, quality=85)
        jpeg_data.seek(0)
        return f"data:image/jpeg;base64,{b64encode(jpeg_data.read()).decode('utf-8')}"

    def _get_json(self,
                  ensure_json: bool = False,
                  stream: bool = False,
                  choices: int = 1
                  ) -> dict[str, Any]:
        ret: dict[str, Any] = {
            "messages": [{"role": "system", "content": self.prompt}] + self._sess,
            "model": self.model.value,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "logprobs": self.logprobs,
            "max_tokens": self.max_reply_tokens or INFO_MAP[self.model].max_tokens,
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

    async def _proc_prompt(self, prompt: Prompt) -> str | list[dict]:
        if isinstance(prompt, str):
            return prompt
        ret = list[dict]()
        for seg in prompt:
            if isinstance(seg, Message.TextSegment):
                ret.append(seg.model_dump())
            elif isinstance(seg, Message.ImageSegment):
                img = await self.cache_image_seg(seg)
                ret.append({
                    "type": "image_url",
                    "image_url": {
                        "url": await self.get_b64url_from_seg(seg),
                        "detail": seg.image_url.detail,
                    },
                    "dims": img.size,
                })
            else:
                raise ValueError(f"Invalid segment type: {seg}")

        return ret
