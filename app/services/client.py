import asyncio
import io
import mimetypes
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, cast

import curl_cffi
import orjson
from curl_cffi.requests import AsyncSession
from gemini_webapi import GeminiClient, ModelOutput
from gemini_webapi.client import GRPC, ChatSession, GeminiError, RPCData, parse_file_name
from gemini_webapi.constants import Endpoint, Headers
from loguru import logger

from app.models import Message
from app.server.middleware import get_uploaded_file_metadata, get_uploaded_file_path
from app.utils import g_config
from app.utils.helper import (
    add_tag,
    normalize_llm_text,
    save_file_to_tempfile,
    save_url_to_tempfile,
)

_UNSET = object()


@dataclass(slots=True)
class GeminiUploadedFileRef:
    """Gemini 上游已完成上传的文件引用。"""

    upload_url: str
    filename: str


def _resolve(value: Any, fallback: Any):
    return fallback if value is _UNSET else value


async def _upload_file_compat(
    file: str | Path | bytes | io.BytesIO,
    client: AsyncSession,
    filename: str | None = None,
) -> str:
    """兼容当前上游损坏的文件上传实现。"""
    if isinstance(file, (str, Path)):
        file_path = Path(file)
        if not file_path.is_file():
            raise ValueError(f"{file_path} is not a valid file.")
        filename = filename or file_path.name
        file_content = file_path.read_bytes()
    elif isinstance(file, io.BytesIO):
        file_content = file.getvalue()
        filename = filename or parse_file_name(file)
    elif isinstance(file, bytes):
        file_content = file
        filename = filename or parse_file_name(file)
    else:
        raise ValueError(f"Unsupported file type: {type(file)}")

    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    mp = curl_cffi.CurlMime()
    mp.addpart(name="file", content_type=content_type, filename=filename, data=file_content)
    try:
        response = await client.post(
            url=Endpoint.UPLOAD,
            headers={
                "Referer": Headers.GEMINI.value["Referer"],
                "Origin": Headers.GEMINI.value["Origin"],
                "User-Agent": Headers.GEMINI.value["User-Agent"],
                **Headers.UPLOAD.value,
            },
            multipart=mp,
            allow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    finally:
        mp.close()


class GeminiClientWrapper(GeminiClient):
    """Gemini client with helper methods."""

    def __init__(self, client_id: str, **kwargs):
        super().__init__(**kwargs)
        self.id = client_id

    async def init(
        self,
        timeout: float = cast(float, _UNSET),
        watchdog_timeout: float = cast(float, _UNSET),
        auto_close: bool = False,
        close_delay: float = cast(float, _UNSET),
        auto_refresh: bool = cast(bool, _UNSET),
        refresh_interval: float = cast(float, _UNSET),
        verbose: bool = cast(bool, _UNSET),
    ) -> None:
        """
        Inject default configuration values.
        """
        config = g_config.gemini
        timeout = cast(float, _resolve(timeout, config.timeout))
        watchdog_timeout = cast(float, _resolve(watchdog_timeout, config.watchdog_timeout))
        close_delay = timeout
        auto_refresh = cast(bool, _resolve(auto_refresh, config.auto_refresh))
        refresh_interval = cast(float, _resolve(refresh_interval, config.refresh_interval))
        verbose = cast(bool, _resolve(verbose, config.verbose))

        try:
            await super().init(
                timeout=timeout,
                watchdog_timeout=watchdog_timeout,
                auto_close=auto_close,
                close_delay=close_delay,
                auto_refresh=auto_refresh,
                refresh_interval=refresh_interval,
                verbose=verbose,
            )
        except Exception:
            logger.exception(f"Failed to initialize GeminiClient {self.id}")
            raise

    def running(self) -> bool:
        return self._running

    async def _prepare_file_data(
        self,
        files: list[str | Path | bytes | io.BytesIO | GeminiUploadedFileRef] | None = None,
    ) -> list[list[Any]] | None:
        """统一准备 Gemini 请求所需的文件上传数据。"""
        file_data = None
        if files:
            uploaded_refs = [file for file in files if isinstance(file, GeminiUploadedFileRef)]
            pending_uploads = [file for file in files if not isinstance(file, GeminiUploadedFileRef)]

            await self._batch_execute(
                [
                    RPCData(
                        rpcid=GRPC.BARD_ACTIVITY,
                        payload='[[["bard_activity_enabled"]]]',
                    )
                ]
            )

            file_data = [[[ref.upload_url], ref.filename] for ref in uploaded_refs]

            if pending_uploads:
                async with AsyncSession(
                    impersonate="chrome",
                    proxy=self.proxy,
                    cookies=dict(self.cookies),
                    timeout=self.timeout,
                ) as upload_client:
                    uploaded_urls = await asyncio.gather(
                        *(_upload_file_compat(file, upload_client) for file in pending_uploads)
                    )

                file_data.extend(
                    [
                        [[url], parse_file_name(file)]
                        for url, file in zip(uploaded_urls, pending_uploads, strict=False)
                    ]
                )

        return file_data

    async def upload_file_reference(
        self,
        file: str | Path | bytes | io.BytesIO,
        filename: str | None = None,
    ) -> GeminiUploadedFileRef:
        """先把文件上传到 Gemini, 再返回可复用的文件引用。"""
        async with AsyncSession(
            impersonate="chrome",
            proxy=self.proxy,
            cookies=dict(self.cookies),
            timeout=self.timeout,
        ) as upload_client:
            upload_url = await _upload_file_compat(file, upload_client, filename=filename)

        resolved_name = filename
        if resolved_name is None:
            if isinstance(file, (str, Path)):
                resolved_name = Path(file).name
            else:
                resolved_name = parse_file_name(file)

        return GeminiUploadedFileRef(upload_url=upload_url, filename=resolved_name)

    async def generate_content(
        self,
        prompt: str,
        files: list[str | Path | bytes | io.BytesIO | GeminiUploadedFileRef] | None = None,
        model: Any = None,
        gem: Any = None,
        chat: ChatSession | None = None,
        temporary: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """
        兼容上游 gemini_webapi 当前版本的文件上传签名变更。
        """
        if self.auto_close:
            await self.reset_close_task()

        if not (isinstance(chat, ChatSession) and chat.cid):
            self._reqid = random.randint(10000, 99999)

        file_data = await self._prepare_file_data(files)

        try:
            await self._batch_execute(
                [
                    RPCData(
                        rpcid=GRPC.BARD_ACTIVITY,
                        payload='[[["bard_activity_enabled"]]]',
                    )
                ]
            )

            session_state = {
                "last_texts": {},
                "last_thoughts": {},
                "last_progress_time": time.time(),
            }
            output = None
            async for generated_output in self._generate(
                prompt=prompt,
                req_file_data=file_data,
                model=model,
                gem=gem,
                chat=chat,
                temporary=temporary,
                session_state=session_state,
                **kwargs,
            ):
                output = generated_output

            if output is None:
                raise GeminiError("Failed to generate contents. No output data found in response.")

            if isinstance(chat, ChatSession):
                output.metadata = chat.metadata
                chat.last_output = output

            return output
        finally:
            if files:
                for file in files:
                    if isinstance(file, io.BytesIO):
                        file.close()

    async def generate_content_stream(
        self,
        prompt: str,
        files: list[str | Path | bytes | io.BytesIO | GeminiUploadedFileRef] | None = None,
        model: Any = None,
        gem: Any = None,
        chat: ChatSession | None = None,
        temporary: bool = False,
        **kwargs,
    ) -> AsyncGenerator[ModelOutput, None]:
        """
        为流式接口补齐与非流式一致的文件上传兼容逻辑。
        """
        if self.auto_close:
            await self.reset_close_task()

        if not (isinstance(chat, ChatSession) and chat.cid):
            self._reqid = random.randint(10000, 99999)

        file_data = await self._prepare_file_data(files)

        try:
            await self._batch_execute(
                [
                    RPCData(
                        rpcid=GRPC.BARD_ACTIVITY,
                        payload='[[["bard_activity_enabled"]]]',
                    )
                ]
            )

            session_state = {
                "last_texts": {},
                "last_thoughts": {},
                "last_progress_time": time.time(),
            }
            output = None
            async for generated_output in self._generate(
                prompt=prompt,
                req_file_data=file_data,
                model=model,
                gem=gem,
                chat=chat,
                temporary=temporary,
                session_state=session_state,
                **kwargs,
            ):
                output = generated_output
                yield generated_output

            if output and isinstance(chat, ChatSession):
                output.metadata = chat.metadata
                chat.last_output = output
        finally:
            if files:
                for file in files:
                    if isinstance(file, io.BytesIO):
                        file.close()

    @staticmethod
    async def process_message(
        message: Message, tempdir: Path | None = None, tagged: bool = True, wrap_tool: bool = True
    ) -> tuple[str, list[Path | str | GeminiUploadedFileRef]]:
        """
        Process a Message into Gemini API format using the PascalCase technical protocol.
        Extracts text, handles files, and appends ToolCalls/ToolResults blocks.
        """
        files: list[Path | str | GeminiUploadedFileRef] = []
        text_fragments: list[str] = []

        if isinstance(message.content, str):
            if message.content or message.role == "tool":
                text_fragments.append(message.content or "")
        elif isinstance(message.content, list):
            for item in message.content:
                if item.type == "text":
                    if item.text or message.role == "tool":
                        text_fragments.append(item.text or "")
                elif item.type == "image_url":
                    if not item.image_url:
                        raise ValueError("Image URL cannot be empty")
                    if url := item.image_url.get("url", None):
                        files.append(await save_url_to_tempfile(url, tempdir))
                    else:
                        raise ValueError("Image URL must contain 'url' key")
                elif item.type == "file":
                    if not item.file:
                        raise ValueError("File cannot be empty")
                    if file_id := item.file.get("file_id", None):
                        metadata = get_uploaded_file_metadata(file_id)
                        upload_url = metadata.get("gemini_file_url")
                        filename = metadata.get("filename")
                        if isinstance(upload_url, str) and upload_url:
                            files.append(
                                GeminiUploadedFileRef(
                                    upload_url=upload_url,
                                    filename=filename if isinstance(filename, str) else file_id,
                                )
                            )
                        else:
                            files.append(get_uploaded_file_path(file_id))
                    elif file_data := item.file.get("file_data", None):
                        filename = item.file.get("filename", "")
                        files.append(await save_file_to_tempfile(file_data, filename, tempdir))
                    elif url := item.file.get("url", None):
                        files.append(await save_url_to_tempfile(url, tempdir))
                    else:
                        raise ValueError("File must contain 'file_id', 'file_data' or 'url' key")
        elif message.content is None and message.role == "tool":
            text_fragments.append("")
        elif message.content is not None:
            raise ValueError("Unsupported message content type.")

        if message.role == "tool":
            tool_name = message.name or "unknown"
            combined_content = "\n".join(text_fragments).strip()
            res_block = (
                f"[Result:{tool_name}]\n[ToolResult]\n{combined_content}\n[/ToolResult]\n[/Result]"
            )
            if wrap_tool:
                text_fragments = [f"[ToolResults]\n{res_block}\n[/ToolResults]"]
            else:
                text_fragments = [res_block]

        if message.tool_calls:
            tool_blocks: list[str] = []
            for call in message.tool_calls:
                params_text = call.function.arguments.strip()
                formatted_params = ""
                if params_text:
                    try:
                        parsed_params = orjson.loads(params_text)
                        if isinstance(parsed_params, dict):
                            for k, v in parsed_params.items():
                                val_str = (
                                    v if isinstance(v, str) else orjson.dumps(v).decode("utf-8")
                                )
                                formatted_params += (
                                    f"[CallParameter:{k}]\n```\n{val_str}\n```\n[/CallParameter]\n"
                                )
                        else:
                            formatted_params += f"```\n{params_text}\n```\n"
                    except orjson.JSONDecodeError:
                        formatted_params += f"```\n{params_text}\n```\n"

                tool_blocks.append(f"[Call:{call.function.name}]\n{formatted_params}[/Call]")

            if tool_blocks:
                tool_section = "[ToolCalls]\n" + "\n".join(tool_blocks) + "\n[/ToolCalls]"
                text_fragments.append(tool_section)

        model_input = "\n".join(fragment for fragment in text_fragments if fragment is not None)

        if (model_input or message.role == "tool") and tagged:
            model_input = add_tag(message.role, model_input)

        return model_input, files

    @staticmethod
    async def process_conversation(
        messages: list[Message], tempdir: Path | None = None
    ) -> tuple[str, list[Path | str | GeminiUploadedFileRef]]:
        conversation: list[str] = []
        files: list[Path | str | GeminiUploadedFileRef] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "tool":
                tool_blocks: list[str] = []
                while i < len(messages) and messages[i].role == "tool":
                    part, part_files = await GeminiClientWrapper.process_message(
                        messages[i], tempdir, tagged=False, wrap_tool=False
                    )
                    tool_blocks.append(part)
                    files.extend(part_files)
                    i += 1

                combined_tool_content = "\n".join(tool_blocks)
                wrapped_content = f"[ToolResults]\n{combined_tool_content}\n[/ToolResults]"
                conversation.append(add_tag("tool", wrapped_content))
            else:
                input_part, files_part = await GeminiClientWrapper.process_message(
                    msg, tempdir, tagged=True
                )
                conversation.append(input_part)
                files.extend(files_part)
                i += 1

        conversation.append(add_tag("assistant", "", unclose=True))
        return "\n".join(conversation), files

    @staticmethod
    def extract_output(response: ModelOutput, include_thoughts: bool = True) -> str:
        text = ""
        if include_thoughts and response.thoughts:
            text += f"<think>{response.thoughts}</think>\n"
        if response.text:
            text += response.text
        else:
            text += str(response)

        return normalize_llm_text(text)
