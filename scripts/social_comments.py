"""Сбор комментариев из YouTube, VK и Telegram.

Модуль построен вокруг небольших компонентов с одной ответственностью:

* модели описывают данные и статистику;
* репозиторий отвечает только за сохранение комментариев;
* API-клиенты инкапсулируют HTTP-взаимодействие;
* сборщики реализуют сценарии работы с конкретными платформами;
* приложение координирует сборщики, не зная деталей их реализации;
* фабрика приложения связывает зависимости в одном месте.

Такое разделение упрощает тестирование, замену CSV на другую систему хранения
и добавление новых социальных платформ без изменения основной логики запуска.
"""


import argparse
import asyncio
import csv
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from threading import local
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import requests
from dotenv import load_dotenv
from tqdm import tqdm


LOGGER_NAME = "social_comment_collector"
TELEGRAM_SESSION_NAME = "social_research_session"
DEFAULT_VK_API_VERSION = "5.199"


# ---------------------------------------------------------------------------
# Модели предметной области
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SocialComment:
    """Единое представление комментария независимо от платформы."""

    platform: str
    query: str
    source: str
    source_id: str
    post_id: str
    comment_id: str
    parent_id: str
    author_id: str
    author_name: str
    text: str
    published_at: str
    like_count: Optional[int]
    url: str
    collected_at: str

    @property
    def unique_key(self) -> tuple[str, str, str, str]:
        """Ключ, по которому комментарий считается уникальным."""

        return (
            self.platform,
            self.source_id,
            self.post_id,
            self.comment_id,
        )


@dataclass(frozen=True, slots=True)
class CollectionStats:
    """Количество полученных и действительно сохранённых комментариев."""

    received: int = 0
    saved: int = 0

    def __add__(self, other: CollectionStats) -> CollectionStats:
        return CollectionStats(
            received=self.received + other.received,
            saved=self.saved + other.saved,
        )


@dataclass(frozen=True, slots=True)
class YouTubeSettings:
    """Настройки сбора данных из YouTube."""

    api_key: str
    queries: tuple[str, ...]
    max_videos_per_query: int
    max_comments_per_video: int
    max_workers: int
    collect_replies: bool


@dataclass(frozen=True, slots=True)
class VKSettings:
    """Настройки сбора данных из VK."""

    access_token: str
    api_version: str
    queries: tuple[str, ...]
    groups: tuple[str, ...]
    max_posts_per_group: int
    max_comments_per_post: int


@dataclass(frozen=True, slots=True)
class TelegramSettings:
    """Настройки сбора данных из Telegram."""

    api_id: int
    api_hash: str
    phone: str
    queries: tuple[str, ...]
    channels: tuple[str, ...]
    max_posts_per_channel: int
    max_comments_per_post: int


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    """Полная конфигурация одного запуска приложения."""

    output_path: Path
    youtube: YouTubeSettings | None
    vk: VKSettings | None
    telegram: TelegramSettings | None


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Возвращает текущее время UTC в формате ISO 8601."""

    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: object | None) -> str:
    """Удаляет лишние пробелы и безопасно обрабатывает пустое значение."""

    if text is None:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def split_comma_separated(value: str | None) -> tuple[str, ...]:
    """Преобразует строку ``a,b,c`` в кортеж непустых значений."""

    if not value:
        return ()

    return tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )


def positive(value: int, minimum: int = 1) -> int:
    """Не позволяет передать в API нулевой или отрицательный лимит."""

    return max(minimum, value)


# ---------------------------------------------------------------------------
# Абстракции
# ---------------------------------------------------------------------------


class CommentRepository(Protocol):
    """Контракт хранилища комментариев.

    Сборщики зависят от этого интерфейса, а не от конкретного CSV-файла.
    Поэтому CSV впоследствии можно заменить, например, на PostgreSQL.
    """

    @property
    def output_path(self) -> Path:
        """Путь или идентификатор места хранения результата."""

    @property
    def total_count(self) -> int:
        """Количество уникальных комментариев в хранилище."""

    def append(self, comments: Sequence[SocialComment]) -> int:
        """Сохраняет новые комментарии и возвращает их количество."""


class AsyncCommentCollector(Protocol):
    """Унифицированный интерфейс любого сборщика комментариев."""

    @property
    def name(self) -> str:
        """Человекочитаемое имя платформы."""

    async def collect(self) -> CollectionStats:
        """Выполняет сбор и возвращает статистику запуска."""


class HttpSessionProvider(Protocol):
    """Выдаёт HTTP-сессию для текущего потока."""

    def get(self) -> requests.Session:
        """Возвращает готовую к использованию HTTP-сессию."""


# ---------------------------------------------------------------------------
# Хранилище
# ---------------------------------------------------------------------------


class CsvCommentRepository:
    """Потокобезопасно и инкрементально сохраняет комментарии в CSV.

    При повторном запуске уже существующий файл просматривается один раз.
    Уникальные ключи загружаются в память, поэтому последующая проверка
    дубликатов выполняется за амортизированное O(1).
    """

    FIELDNAMES = tuple(field.name for field in fields(SocialComment))

    def __init__(
        self,
        output_path: Path,
        logger: logging.Logger,
    ) -> None:
        self._output_path = output_path.expanduser().resolve()
        self._logger = logger
        self._lock = threading.Lock()
        self._saved_keys: set[tuple[str, str, str, str]] = set()

        self._prepare_parent_directory()
        self._load_existing_keys()

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def total_count(self) -> int:
        return len(self._saved_keys)

    def _prepare_parent_directory(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key_from_row(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("platform", "")),
            str(row.get("source_id", "")),
            str(row.get("post_id", "")),
            str(row.get("comment_id", "")),
        )

    def _load_existing_keys(self) -> None:
        if not self._output_path.exists() or self._output_path.stat().st_size == 0:
            return

        try:
            with self._output_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                for row in csv.DictReader(file):
                    self._saved_keys.add(self._key_from_row(row))

            self._logger.info(
                "Найден существующий файл %s. Загружено уникальных "
                "комментариев: %s",
                self._output_path,
                self.total_count,
            )
        except (OSError, csv.Error) as exc:
            self._logger.warning(
                "Не удалось прочитать существующий CSV %s: %s",
                self._output_path,
                exc,
            )

    def append(self, comments: Sequence[SocialComment]) -> int:
        if not comments:
            return 0

        # Одна блокировка защищает и набор ключей, и запись в файл. Это важно,
        # потому что YouTube обрабатывает несколько видео параллельно.
        with self._lock:
            rows: list[dict[str, Any]] = []
            new_keys: set[tuple[str, str, str, str]] = set()

            for comment in comments:
                normalized_text = normalize_text(comment.text)
                if not normalized_text:
                    continue

                key = comment.unique_key
                if key in self._saved_keys or key in new_keys:
                    continue

                row = asdict(comment)
                row["text"] = normalized_text
                rows.append(row)
                new_keys.add(key)

            if not rows:
                return 0

            file_has_data = (
                self._output_path.exists()
                and self._output_path.stat().st_size > 0
            )

            # Ключи добавляются в память только после успешной записи. Так при
            # ошибке файловой системы данные не будут ошибочно считаться сохранёнными.
            with self._output_path.open(
                "a",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.FIELDNAMES,
                    extrasaction="ignore",
                )

                if not file_has_data:
                    writer.writeheader()

                writer.writerows(rows)
                file.flush()
                os.fsync(file.fileno())

            self._saved_keys.update(new_keys)
            return len(rows)


# ---------------------------------------------------------------------------
# HTTP-инфраструктура и ошибки внешних API
# ---------------------------------------------------------------------------


class ExternalApiError(RuntimeError):
    """Базовая ошибка при взаимодействии с внешним API."""


class YouTubeQuotaExceeded(ExternalApiError):
    """Квота YouTube Data API закончилась."""


class YouTubeCommentsDisabled(ExternalApiError):
    """Комментарии к конкретному видео отключены."""


class ThreadLocalRequestsSessionProvider:
    """Создаёт отдельную ``requests.Session`` для каждого рабочего потока."""

    def __init__(
        self,
        pool_size: int = 10,
        retries: int = 3,
    ) -> None:
        self._pool_size = pool_size
        self._retries = retries
        self._thread_local = local()

    def get(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            return session

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self._pool_size,
            pool_maxsize=self._pool_size,
            max_retries=self._retries,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self._thread_local.session = session
        return session


class YouTubeApiClient:
    """Низкоуровневый клиент YouTube Data API v3."""

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(
        self,
        api_key: str,
        session_provider: HttpSessionProvider,
        timeout_seconds: int = 45,
    ) -> None:
        if not api_key:
            raise ValueError("Не найден YOUTUBE_API_KEY")

        self._api_key = api_key
        self._session_provider = session_provider
        self._timeout_seconds = timeout_seconds

    def get(
        self,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        response = self._session_provider.get().get(
            f"{self.BASE_URL}/{endpoint}",
            params=dict(params),
            headers={"X-Goog-Api-Key": self._api_key},
            timeout=self._timeout_seconds,
        )

        if response.ok:
            return response.json()

        error_data = self._read_error_payload(response)
        error = error_data.get("error", {})
        message = str(error.get("message", ""))
        status = str(error.get("status", ""))
        reasons = {
            str(item.get("reason", ""))
            for item in error.get("errors", [])
        }

        if self._is_quota_error(
            response.status_code,
            message,
            status,
            reasons,
        ):
            raise YouTubeQuotaExceeded(
                f"HTTP {response.status_code}: {message}"
            )

        if "commentsDisabled" in reasons:
            raise YouTubeCommentsDisabled(message or "Комментарии отключены")

        raise ExternalApiError(
            f"YouTube API: endpoint={endpoint}, "
            f"HTTP={response.status_code}, reasons={sorted(reasons)}, "
            f"message={message}"
        )

    @staticmethod
    def _read_error_payload(response: requests.Response) -> dict[str, Any]:
        try:
            return response.json()
        except ValueError:
            return {
                "error": {
                    "code": response.status_code,
                    "message": response.text[:500],
                }
            }

    @staticmethod
    def _is_quota_error(
        status_code: int,
        message: str,
        status: str,
        reasons: set[str],
    ) -> bool:
        quota_markers = {
            "quotaExceeded",
            "dailyLimitExceeded",
            "rateLimitExceeded",
        }
        lower_message = message.lower()

        return (
            bool(reasons.intersection(quota_markers))
            or status == "RESOURCE_EXHAUSTED"
            or "quota exceeded" in lower_message
            or (status_code == 429 and "quota" in lower_message)
        )


class VKApiClient:
    """Низкоуровневый клиент VK API."""

    BASE_URL = "https://api.vk.com/method"

    def __init__(
        self,
        access_token: str,
        api_version: str,
        session: requests.Session,
        timeout_seconds: int = 30,
    ) -> None:
        if not access_token:
            raise ValueError("Не найден VK_ACCESS_TOKEN")

        self._access_token = access_token
        self._api_version = api_version
        self._session = session
        self._timeout_seconds = timeout_seconds

    def get(
        self,
        method: str,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_params = {
            **params,
            "access_token": self._access_token,
            "v": self._api_version,
        }

        response = self._session.get(
            f"{self.BASE_URL}/{method}",
            params=request_params,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        if "error" in data:
            error = data["error"]
            raise ExternalApiError(
                f"VK API {error.get('error_code')}: "
                f"{error.get('error_msg')}"
            )

        return data.get("response", {})


# ---------------------------------------------------------------------------
# Преобразование ответов API в доменную модель
# ---------------------------------------------------------------------------


class YouTubeCommentMapper:
    """Преобразует JSON YouTube в ``SocialComment``."""

    @staticmethod
    def top_level(
        query: str,
        video_id: str,
        item: Mapping[str, Any],
    ) -> SocialComment | None:
        top_level = item.get("snippet", {}).get("topLevelComment", {})
        snippet = top_level.get("snippet", {})
        comment_id = str(top_level.get("id", ""))

        if not comment_id:
            return None

        return YouTubeCommentMapper._build(
            query=query,
            video_id=video_id,
            comment_id=comment_id,
            parent_id="",
            snippet=snippet,
        )

    @staticmethod
    def reply(
        query: str,
        video_id: str,
        parent_id: str,
        item: Mapping[str, Any],
    ) -> SocialComment | None:
        snippet = item.get("snippet", {})
        comment_id = str(item.get("id", ""))

        if not comment_id:
            return None

        return YouTubeCommentMapper._build(
            query=query,
            video_id=video_id,
            comment_id=comment_id,
            parent_id=parent_id,
            snippet=snippet,
        )

    @staticmethod
    def _build(
        query: str,
        video_id: str,
        comment_id: str,
        parent_id: str,
        snippet: Mapping[str, Any],
    ) -> SocialComment:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        return SocialComment(
            platform="youtube",
            query=query,
            source=video_url,
            source_id=video_id,
            post_id=video_id,
            comment_id=comment_id,
            parent_id=parent_id,
            author_id=str(
                snippet.get("authorChannelId", {}).get("value", "")
            ),
            author_name=str(snippet.get("authorDisplayName", "")),
            text=normalize_text(snippet.get("textDisplay")),
            published_at=str(snippet.get("publishedAt", "")),
            like_count=snippet.get("likeCount"),
            url=f"{video_url}&lc={comment_id}",
            collected_at=utc_now_iso(),
        )


class VKCommentMapper:
    """Преобразует JSON VK в ``SocialComment``."""

    @staticmethod
    def from_item(
        query: str,
        owner_id: int,
        post_id: int,
        item: Mapping[str, Any],
    ) -> SocialComment | None:
        comment_id = str(item.get("id", ""))
        if not comment_id:
            return None

        timestamp = item.get("date", 0)
        post_url = f"https://vk.com/wall{owner_id}_{post_id}"

        return SocialComment(
            platform="vk",
            query=query,
            source=post_url,
            source_id=str(owner_id),
            post_id=str(post_id),
            comment_id=comment_id,
            parent_id=str(item.get("reply_to_comment", "")),
            author_id=str(item.get("from_id", "")),
            author_name="",
            text=normalize_text(item.get("text")),
            published_at=(
                datetime.fromtimestamp(
                    timestamp,
                    tz=timezone.utc,
                ).isoformat()
                if timestamp
                else ""
            ),
            like_count=item.get("likes", {}).get("count"),
            url=f"{post_url}?reply={comment_id}",
            collected_at=utc_now_iso(),
        )


class TelegramCommentMapper:
    """Преобразует объект сообщения Telethon в ``SocialComment``."""

    @staticmethod
    def from_message(
        query: str,
        channel: str,
        source_id: str,
        post_id: str,
        message: Any,
    ) -> SocialComment:
        channel_name = channel.replace("@", "").strip("/")

        return SocialComment(
            platform="telegram",
            query=query,
            source=channel,
            source_id=source_id,
            post_id=post_id,
            comment_id=str(message.id),
            parent_id=post_id,
            author_id=str(message.sender_id or ""),
            author_name="",
            text=normalize_text(message.message),
            published_at=(
                message.date.isoformat()
                if message.date
                else ""
            ),
            like_count=None,
            url=(
                f"https://t.me/{channel_name}/{post_id}"
                f"?comment={message.id}"
            ),
            collected_at=utc_now_iso(),
        )


# ---------------------------------------------------------------------------
# Сборщики платформ
# ---------------------------------------------------------------------------


class YouTubeCollector:
    """Оркестрирует поиск видео и загрузку комментариев YouTube."""

    def __init__(
        self,
        api_client: YouTubeApiClient,
        repository: CommentRepository,
        settings: YouTubeSettings,
        logger: logging.Logger,
    ) -> None:
        self._api = api_client
        self._repository = repository
        self._settings = settings
        self._logger = logger

    def collect(self) -> CollectionStats:
        total = CollectionStats()

        for index, query in enumerate(self._settings.queries, start=1):
            self._logger.info(
                "YouTube: запрос %s/%s — %s",
                index,
                len(self._settings.queries),
                query,
            )

            try:
                video_ids = self._search_video_ids(query)
            except YouTubeQuotaExceeded as exc:
                self._logger.error("YouTube: квота исчерпана: %s", exc)
                break
            except Exception as exc:
                self._logger.exception(
                    "YouTube: ошибка поиска по запросу %r: %s",
                    query,
                    exc,
                )
                continue

            if not video_ids:
                self._logger.info("YouTube: видео не найдены")
                continue

            self._logger.info(
                "YouTube: найдено видео: %s; рабочих потоков: %s",
                len(video_ids),
                self._settings.max_workers,
            )

            query_stats, quota_exceeded = self._collect_videos_concurrently(
                query,
                video_ids,
            )
            total = total + query_stats

            if quota_exceeded:
                self._logger.warning(
                    "YouTube: сбор остановлен из-за исчерпания квоты. "
                    "Все ранее полученные данные уже сохранены."
                )
                break

        return total

    def _search_video_ids(self, query: str) -> list[str]:
        video_ids: list[str] = []
        page_token: str | None = None

        while len(video_ids) < self._settings.max_videos_per_query:
            params: dict[str, Any] = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": min(
                    50,
                    self._settings.max_videos_per_query - len(video_ids),
                ),
                "order": "relevance",
            }
            if page_token:
                params["pageToken"] = page_token

            data = self._api.get("search", params)
            for item in data.get("items", []):
                video_id = item.get("id", {}).get("videoId")
                if video_id and video_id not in video_ids:
                    video_ids.append(str(video_id))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return video_ids

    def _collect_videos_concurrently(
        self,
        query: str,
        video_ids: Sequence[str],
    ) -> tuple[CollectionStats, bool]:
        total = CollectionStats()
        quota_exceeded = False

        with ThreadPoolExecutor(
            max_workers=self._settings.max_workers
        ) as executor:
            future_to_video = {
                executor.submit(
                    self._collect_video_comments,
                    query,
                    video_id,
                ): video_id
                for video_id in video_ids
            }

            progress = tqdm(
                as_completed(future_to_video),
                total=len(future_to_video),
                desc=f"YouTube: {query}",
            )

            for future in progress:
                video_id = future_to_video[future]

                try:
                    stats = future.result()
                    total = total + stats
                    tqdm.write(
                        f"YouTube: {video_id} — получено "
                        f"{stats.received}, сохранено {stats.saved}"
                    )
                except YouTubeQuotaExceeded as exc:
                    self._logger.error(
                        "YouTube: квота исчерпана при обработке %s: %s",
                        video_id,
                        exc,
                    )
                    quota_exceeded = True

                    # Уже запущенные запросы завершатся, но ещё не начавшиеся
                    # задачи не должны расходовать оставшуюся квоту.
                    for pending_future in future_to_video:
                        pending_future.cancel()
                    break
                except Exception as exc:
                    self._logger.exception(
                        "YouTube: ошибка video_id=%s: %s",
                        video_id,
                        exc,
                    )

        return total, quota_exceeded

    def _collect_video_comments(
        self,
        query: str,
        video_id: str,
    ) -> CollectionStats:
        total = CollectionStats()
        page_token: str | None = None
        top_level_received = 0

        while top_level_received < self._settings.max_comments_per_video:
            params: dict[str, Any] = {
                "part": "snippet,replies",
                "videoId": video_id,
                "maxResults": min(
                    100,
                    self._settings.max_comments_per_video
                    - top_level_received,
                ),
                "textFormat": "plainText",
                "order": "relevance",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                data = self._api.get("commentThreads", params)
            except YouTubeCommentsDisabled:
                self._logger.info(
                    "YouTube: комментарии отключены для video_id=%s",
                    video_id,
                )
                break

            page_comments = [
                comment
                for item in data.get("items", [])
                if (
                    comment := YouTubeCommentMapper.top_level(
                        query,
                        video_id,
                        item,
                    )
                )
                is not None
            ]

            saved = self._repository.append(page_comments)
            page_stats = CollectionStats(
                received=len(page_comments),
                saved=saved,
            )
            total = total + page_stats
            top_level_received += len(page_comments)

            if self._settings.collect_replies:
                for item in data.get("items", []):
                    parent_id = str(
                        item.get("snippet", {})
                        .get("topLevelComment", {})
                        .get("id", "")
                    )
                    total_replies = int(
                        item.get("snippet", {}).get("totalReplyCount", 0)
                    )

                    if parent_id and total_replies > 0:
                        total = total + self._collect_replies(
                            query=query,
                            video_id=video_id,
                            parent_id=parent_id,
                            max_replies=min(100, total_replies),
                        )

            page_token = data.get("nextPageToken")
            if not page_token or not page_comments:
                break

        return total

    def _collect_replies(
        self,
        query: str,
        video_id: str,
        parent_id: str,
        max_replies: int,
    ) -> CollectionStats:
        total = CollectionStats()
        page_token: str | None = None

        while total.received < max_replies:
            params: dict[str, Any] = {
                "part": "snippet",
                "parentId": parent_id,
                "maxResults": min(100, max_replies - total.received),
                "textFormat": "plainText",
            }
            if page_token:
                params["pageToken"] = page_token

            data = self._api.get("comments", params)
            page_comments = [
                comment
                for item in data.get("items", [])
                if (
                    comment := YouTubeCommentMapper.reply(
                        query,
                        video_id,
                        parent_id,
                        item,
                    )
                )
                is not None
            ]

            saved = self._repository.append(page_comments)
            total = total + CollectionStats(
                received=len(page_comments),
                saved=saved,
            )

            page_token = data.get("nextPageToken")
            if not page_token or not page_comments:
                break

        return total


class VKCollector:
    """Ищет подходящие посты VK и сохраняет их комментарии."""

    def __init__(
        self,
        api_client: VKApiClient,
        repository: CommentRepository,
        settings: VKSettings,
        logger: logging.Logger,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api = api_client
        self._repository = repository
        self._settings = settings
        self._logger = logger
        self._sleep = sleep

    def collect(self) -> CollectionStats:
        total = CollectionStats()

        for group in self._settings.groups:
            try:
                owner_id = self._resolve_owner_id(group)
            except Exception as exc:
                self._logger.exception(
                    "VK: не удалось определить группу %s: %s",
                    group,
                    exc,
                )
                continue

            if owner_id is None:
                self._logger.warning(
                    "VK: не удалось определить owner_id для %s",
                    group,
                )
                continue

            self._logger.info(
                "VK: группа %s, owner_id=%s",
                group,
                owner_id,
            )

            try:
                posts = self._get_wall_posts(owner_id)
            except Exception as exc:
                self._logger.exception(
                    "VK: не удалось получить посты группы %s: %s",
                    group,
                    exc,
                )
                continue

            for post in tqdm(posts, desc=f"VK: {group}"):
                post_id = post.get("id")
                if post_id is None:
                    continue

                post_text = normalize_text(post.get("text")).casefold()
                matched_queries = [
                    query
                    for query in self._settings.queries
                    if query.casefold() in post_text
                ]

                for query in matched_queries:
                    total = total + self._collect_post_comments(
                        query=query,
                        owner_id=owner_id,
                        post_id=int(post_id),
                    )

        return total

    def _resolve_owner_id(self, screen_name: str) -> int | None:
        cleaned_name = (
            screen_name.replace("https://vk.com/", "")
            .replace("http://vk.com/", "")
            .strip("/")
        )

        response = self._api.get(
            "utils.resolveScreenName",
            {"screen_name": cleaned_name},
        )
        object_type = response.get("type")
        object_id = response.get("object_id")

        if object_id is None:
            return None
        if object_type in {"group", "page"}:
            return -int(object_id)
        if object_type == "user":
            return int(object_id)
        return None

    def _get_wall_posts(self, owner_id: int) -> list[dict[str, Any]]:
        posts: list[dict[str, Any]] = []
        offset = 0

        while len(posts) < self._settings.max_posts_per_group:
            response = self._api.get(
                "wall.get",
                {
                    "owner_id": owner_id,
                    "offset": offset,
                    "count": min(
                        100,
                        self._settings.max_posts_per_group - len(posts),
                    ),
                    "filter": "owner",
                },
            )
            items = response.get("items", [])
            if not items:
                break

            posts.extend(items)
            offset += len(items)
            self._sleep(0.35)

        return posts

    def _collect_post_comments(
        self,
        query: str,
        owner_id: int,
        post_id: int,
    ) -> CollectionStats:
        total = CollectionStats()
        offset = 0

        while total.received < self._settings.max_comments_per_post:
            try:
                response = self._api.get(
                    "wall.getComments",
                    {
                        "owner_id": owner_id,
                        "post_id": post_id,
                        "offset": offset,
                        "count": min(
                            100,
                            self._settings.max_comments_per_post
                            - total.received,
                        ),
                        "need_likes": 1,
                        "sort": "asc",
                        "extended": 0,
                    },
                )
            except Exception as exc:
                self._logger.warning(
                    "VK: комментарии недоступны для owner_id=%s, "
                    "post_id=%s: %s",
                    owner_id,
                    post_id,
                    exc,
                )
                break

            items = response.get("items", [])
            if not items:
                break

            page_comments = [
                comment
                for item in items
                if (
                    comment := VKCommentMapper.from_item(
                        query,
                        owner_id,
                        post_id,
                        item,
                    )
                )
                is not None
            ]

            saved = self._repository.append(page_comments)
            total = total + CollectionStats(
                received=len(page_comments),
                saved=saved,
            )
            offset += len(items)
            self._sleep(0.35)

        return total


class TelegramCollector:
    """Асинхронно собирает комментарии к найденным постам Telegram."""

    name = "Telegram"
    BATCH_SIZE = 50

    def __init__(
        self,
        client: Any,
        repository: CommentRepository,
        settings: TelegramSettings,
        logger: logging.Logger,
    ) -> None:
        self._client = client
        self._repository = repository
        self._settings = settings
        self._logger = logger

    async def collect(self) -> CollectionStats:
        total = CollectionStats()
        await self._client.start(phone=self._settings.phone)

        try:
            for channel in self._settings.channels:
                self._logger.info("Telegram: канал или чат %s", channel)

                try:
                    entity = await self._client.get_entity(channel)
                except Exception as exc:
                    self._logger.warning(
                        "Telegram: не удалось открыть %s: %s",
                        channel,
                        exc,
                    )
                    continue

                source_id = str(getattr(entity, "id", ""))

                for query in self._settings.queries:
                    total = total + await self._collect_query_from_channel(
                        entity=entity,
                        source_id=source_id,
                        channel=channel,
                        query=query,
                    )
        finally:
            await self._client.disconnect()

        return total

    async def _collect_query_from_channel(
        self,
        entity: Any,
        source_id: str,
        channel: str,
        query: str,
    ) -> CollectionStats:
        total = CollectionStats()
        self._logger.info(
            "Telegram: поиск постов по запросу %r",
            query,
        )

        try:
            async for post in self._client.iter_messages(
                entity,
                search=query,
                limit=self._settings.max_posts_per_channel,
            ):
                total = total + await self._collect_post_comments(
                    entity=entity,
                    source_id=source_id,
                    channel=channel,
                    query=query,
                    post_id=str(post.id),
                )
        except Exception as exc:
            self._logger.warning(
                "Telegram: ошибка поиска %r в %s: %s",
                query,
                channel,
                exc,
            )

        return total

    async def _collect_post_comments(
        self,
        entity: Any,
        source_id: str,
        channel: str,
        query: str,
        post_id: str,
    ) -> CollectionStats:
        total = CollectionStats()
        batch: list[SocialComment] = []

        try:
            async for message in self._client.iter_messages(
                entity,
                reply_to=int(post_id),
                limit=self._settings.max_comments_per_post,
            ):
                batch.append(
                    TelegramCommentMapper.from_message(
                        query=query,
                        channel=channel,
                        source_id=source_id,
                        post_id=post_id,
                        message=message,
                    )
                )

                if len(batch) >= self.BATCH_SIZE:
                    total = total + self._save_batch(batch)
                    batch.clear()

            if batch:
                total = total + self._save_batch(batch)
        except Exception as exc:
            self._logger.warning(
                "Telegram: комментарии недоступны для post_id=%s: %s",
                post_id,
                exc,
            )

        return total

    def _save_batch(
        self,
        comments: Sequence[SocialComment],
    ) -> CollectionStats:
        saved = self._repository.append(comments)
        return CollectionStats(received=len(comments), saved=saved)


# ---------------------------------------------------------------------------
# Адаптеры и приложение
# ---------------------------------------------------------------------------


class SyncCollectorAdapter:
    """Адаптирует синхронный сборщик к общему асинхронному интерфейсу."""

    def __init__(
        self,
        name: str,
        collect_function: Callable[[], CollectionStats],
    ) -> None:
        self._name = name
        self._collect_function = collect_function

    @property
    def name(self) -> str:
        return self._name

    async def collect(self) -> CollectionStats:
        # Синхронные requests-вызовы выполняются вне event loop, поэтому
        # асинхронная работа Telegram не блокируется архитектурно.
        return await asyncio.to_thread(self._collect_function)


class SocialCommentApplication:
    """Запускает зарегистрированные сборщики и печатает общий итог."""

    def __init__(
        self,
        collectors: Sequence[AsyncCommentCollector],
        repository: CommentRepository,
        logger: logging.Logger,
    ) -> None:
        self._collectors = collectors
        self._repository = repository
        self._logger = logger

    async def run(self) -> CollectionStats:
        total = CollectionStats()

        if not self._collectors:
            self._logger.warning(
                "Не выбрана ни одна платформа для сбора комментариев"
            )

        # Платформы запускаются последовательно, чтобы их прогресс-бары и
        # сообщения не перемешивались. Каждый сборщик внутри может применять
        # собственную модель конкурентности.
        for collector in self._collectors:
            try:
                stats = await collector.collect()
                total = total + stats
                self._logger.info(
                    "%s: получено %s, сохранено %s",
                    collector.name,
                    stats.received,
                    stats.saved,
                )
            except Exception as exc:
                self._logger.exception(
                    "%s: критическая ошибка: %s",
                    collector.name,
                    exc,
                )

        self._logger.info("Сбор завершён")
        self._logger.info(
            "Получено комментариев в текущем запуске: %s",
            total.received,
        )
        self._logger.info(
            "Сохранено новых комментариев: %s",
            total.saved,
        )
        self._logger.info(
            "Уникальных комментариев в хранилище: %s",
            self._repository.total_count,
        )
        self._logger.info(
            "Файл результата: %s",
            self._repository.output_path,
        )

        return total


# ---------------------------------------------------------------------------
# Разбор настроек и сборка зависимостей
# ---------------------------------------------------------------------------


class SettingsLoader:
    """Преобразует аргументы CLI и переменные окружения в конфигурацию."""

    @staticmethod
    def from_namespace(args: argparse.Namespace) -> ApplicationSettings:
        queries = split_comma_separated(args.queries)
        if not queries:
            raise ValueError("Не передано ни одного поискового запроса")

        tg_channels = split_comma_separated(args.tg_channels)
        vk_groups = split_comma_separated(args.vk_groups)

        youtube = None
        if args.youtube:
            youtube = YouTubeSettings(
                api_key=os.getenv("YOUTUBE_API_KEY", ""),
                queries=queries,
                max_videos_per_query=positive(args.max_videos),
                max_comments_per_video=positive(args.max_comments),
                max_workers=positive(args.workers),
                collect_replies=args.youtube_replies,
            )

        vk = None
        if vk_groups:
            vk = VKSettings(
                access_token=os.getenv("VK_ACCESS_TOKEN", ""),
                api_version=os.getenv(
                    "VK_API_VERSION",
                    DEFAULT_VK_API_VERSION,
                ),
                queries=queries,
                groups=vk_groups,
                max_posts_per_group=positive(args.max_posts),
                max_comments_per_post=positive(args.max_comments),
            )

        telegram = None
        if tg_channels:
            telegram = TelegramSettings(
                api_id=SettingsLoader._read_int_env("TELEGRAM_API_ID"),
                api_hash=os.getenv("TELEGRAM_API_HASH", ""),
                phone=os.getenv("TELEGRAM_PHONE", ""),
                queries=queries,
                channels=tg_channels,
                max_posts_per_channel=positive(args.max_posts),
                max_comments_per_post=positive(args.max_comments),
            )

        return ApplicationSettings(
            output_path=Path(args.out),
            youtube=youtube,
            vk=vk,
            telegram=telegram,
        )

    @staticmethod
    def _read_int_env(name: str) -> int:
        raw_value = os.getenv(name, "0")
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Переменная {name} должна содержать целое число"
            ) from exc


class ApplicationFactory:
    """Создаёт объекты приложения и связывает их зависимости."""

    @staticmethod
    def create(
        settings: ApplicationSettings,
        logger: logging.Logger,
    ) -> SocialCommentApplication:
        repository = CsvCommentRepository(
            output_path=settings.output_path,
            logger=logger,
        )
        collectors: list[AsyncCommentCollector] = []

        if settings.youtube is not None:
            ApplicationFactory._register_youtube(
                collectors,
                repository,
                settings.youtube,
                logger,
            )

        if settings.vk is not None:
            ApplicationFactory._register_vk(
                collectors,
                repository,
                settings.vk,
                logger,
            )

        if settings.telegram is not None:
            ApplicationFactory._register_telegram(
                collectors,
                repository,
                settings.telegram,
                logger,
            )

        return SocialCommentApplication(
            collectors=collectors,
            repository=repository,
            logger=logger,
        )

    @staticmethod
    def _register_youtube(
        collectors: list[AsyncCommentCollector],
        repository: CommentRepository,
        settings: YouTubeSettings,
        logger: logging.Logger,
    ) -> None:
        try:
            api_client = YouTubeApiClient(
                api_key=settings.api_key,
                session_provider=ThreadLocalRequestsSessionProvider(
                    pool_size=max(10, settings.max_workers),
                ),
            )
            collector = YouTubeCollector(
                api_client=api_client,
                repository=repository,
                settings=settings,
                logger=logger,
            )
            collectors.append(
                SyncCollectorAdapter("YouTube", collector.collect)
            )
        except Exception as exc:
            logger.error("YouTube: ошибка конфигурации: %s", exc)

    @staticmethod
    def _register_vk(
        collectors: list[AsyncCommentCollector],
        repository: CommentRepository,
        settings: VKSettings,
        logger: logging.Logger,
    ) -> None:
        try:
            api_client = VKApiClient(
                access_token=settings.access_token,
                api_version=settings.api_version,
                session=requests.Session(),
            )
            collector = VKCollector(
                api_client=api_client,
                repository=repository,
                settings=settings,
                logger=logger,
            )
            collectors.append(SyncCollectorAdapter("VK", collector.collect))
        except Exception as exc:
            logger.error("VK: ошибка конфигурации: %s", exc)

    @staticmethod
    def _register_telegram(
        collectors: list[AsyncCommentCollector],
        repository: CommentRepository,
        settings: TelegramSettings,
        logger: logging.Logger,
    ) -> None:
        try:
            if not settings.api_id or not settings.api_hash:
                raise ValueError(
                    "Не найдены TELEGRAM_API_ID / TELEGRAM_API_HASH"
                )

            # Telethon является необязательной зависимостью: импортируем его
            # только когда пользователь действительно включил Telegram.
            from telethon import TelegramClient

            client = TelegramClient(
                TELEGRAM_SESSION_NAME,
                settings.api_id,
                settings.api_hash,
            )
            collectors.append(
                TelegramCollector(
                    client=client,
                    repository=repository,
                    settings=settings,
                    logger=logger,
                )
            )
        except Exception as exc:
            logger.error("Telegram: ошибка конфигурации: %s", exc)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер без запуска бизнес-логики."""

    parser = argparse.ArgumentParser(
        description=(
            "Сбор комментариев из YouTube, VK и Telegram "
            "с немедленным сохранением в CSV."
        )
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="Поисковые запросы через запятую",
    )
    parser.add_argument(
        "--tg-channels",
        default="",
        help="Telegram-каналы или чаты через запятую",
    )
    parser.add_argument(
        "--vk-groups",
        default="",
        help="VK-группы через запятую",
    )
    parser.add_argument(
        "--youtube",
        action="store_true",
        help="Включить сбор комментариев YouTube",
    )
    parser.add_argument(
        "--out",
        default="social_comments.csv",
        help="Путь к итоговому CSV",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=5,
        help="Максимум видео YouTube на один запрос",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=50,
        help="Максимум постов VK или Telegram на источник",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=50,
        help="Максимум комментариев на пост или видео",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Количество рабочих потоков YouTube",
    )
    parser.add_argument(
        "--youtube-replies",
        action="store_true",
        help="Собирать ответы на комментарии YouTube",
    )
    return parser


def configure_logging() -> logging.Logger:
    """Настраивает единый формат сообщений приложения."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(LOGGER_NAME)


async def async_main() -> None:
    """Точка входа, удобная для вызова из тестов и других модулей."""

    load_dotenv()
    logger = configure_logging()
    args = build_argument_parser().parse_args()
    settings = SettingsLoader.from_namespace(args)
    application = ApplicationFactory.create(settings, logger)
    await application.run()


def main() -> None:
    """Синхронная точка входа командной строки."""

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print(
            "\nОстановка пользователем. "
            "Все уже полученные комментарии сохранены."
        )


if __name__ == "__main__":
    main()
