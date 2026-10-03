from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

import aiohttp


class WeatherProvider(Enum):
    OPENWEATHER = "openweather"
    TOMORROW_IO = "tomorrow_io"


class WeatherError(Exception):
    def __init__(self, provider: WeatherProvider, message: str, status_code: Optional[int] = None) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider.value}] {message}")


class WeatherTimeoutError(WeatherError):
    pass


class WeatherAuthError(WeatherError):
    pass


class WeatherRateLimitError(WeatherError):
    pass


@dataclass(frozen=True)
class CurrentConditions:
    provider: WeatherProvider
    location_name: str
    latitude: float
    longitude: float
    temperature_celsius: float
    feels_like_celsius: float
    humidity_percent: float
    pressure_hpa: float
    wind_speed_mps: float
    wind_direction_deg: Optional[float]
    cloud_cover_percent: Optional[float]
    precipitation_mm: Optional[float]
    weather_code: Optional[str]
    description: str
    observed_at_epoch: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ForecastPeriod:
    start_epoch: float
    end_epoch: float
    temperature_celsius: float
    temperature_min_celsius: Optional[float]
    temperature_max_celsius: Optional[float]
    humidity_percent: Optional[float]
    precipitation_probability_percent: Optional[float]
    precipitation_mm: Optional[float]
    wind_speed_mps: Optional[float]
    weather_code: Optional[str]
    description: str


@dataclass(frozen=True)
class Forecast:
    provider: WeatherProvider
    location_name: str
    latitude: float
    longitude: float
    periods: tuple[ForecastPeriod, ...]
    fetched_at_epoch: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderHealth:
    provider: WeatherProvider
    status: HealthStatus
    latency_seconds: Optional[float]
    last_checked_epoch: float
    detail: str = ""


@dataclass
class _CacheEntry:
    value: Any
    expires_at_epoch: float


class TTLCache:
    def __init__(self, default_ttl_seconds: float = 300.0, max_entries: int = 512) -> None:
        self._default_ttl_seconds = default_ttl_seconds
        self._max_entries = max_entries
        self._store: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at_epoch < time.monotonic():
                del self._store[key]
                return None
            return entry.value

    async def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        async with self._lock:
            if len(self._store) >= self._max_entries and key not in self._store:
                oldest_key = min(self._store, key=lambda k: self._store[k].expires_at_epoch, default=None)
                if oldest_key is not None:
                    del self._store[oldest_key]
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
            self._store[key] = _CacheEntry(value=value, expires_at_epoch=time.monotonic() + ttl)

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()


class WeatherClient(Protocol):
    async def get_current(self, latitude: float, longitude: float) -> CurrentConditions: ...

    async def get_forecast(self, latitude: float, longitude: float, hours: int) -> Forecast: ...

    async def health_check(self) -> ProviderHealth: ...


class BaseWeatherClient:
    provider: WeatherProvider

    def __init__(
        self,
        api_key: str,
        session: Optional[aiohttp.ClientSession] = None,
        request_timeout_seconds: float = 10.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._request_timeout_seconds = request_timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    async def __aenter__(self) -> "BaseWeatherClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        session = await self._ensure_session()
        last_error: Optional[Exception] = None
        timeout = aiohttp.ClientTimeout(total=self._request_timeout_seconds)

        for attempt in range(self._max_retries + 1):
            try:
                async with session.get(url, params=params, timeout=timeout) as response:
                    if response.status == 401 or response.status == 403:
                        raise WeatherAuthError(
                            self.provider, "authentication failed", response.status
                        )
                    if response.status == 429:
                        raise WeatherRateLimitError(
                            self.provider, "rate limit exceeded", response.status
                        )
                    if response.status >= 400:
                        body = await response.text()
                        raise WeatherError(
                            self.provider,
                            f"request failed with status {response.status}: {body[:200]}",
                            response.status,
                        )
                    return await response.json()
            except asyncio.TimeoutError:
                last_error = WeatherTimeoutError(self.provider, "request timed out")
            except (WeatherAuthError, WeatherRateLimitError):
                raise
            except aiohttp.ClientError as exc:
                last_error = WeatherError(self.provider, f"client error: {exc}")
            except WeatherError:
                raise

            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_backoff_seconds * (2 ** attempt))

        assert last_error is not None
        raise last_error


class OpenWeatherClient(BaseWeatherClient):
    provider = WeatherProvider.OPENWEATHER
    _BASE_URL = "https://api.openweathermap.org/data/2.5"

    async def get_current(self, latitude: float, longitude: float) -> CurrentConditions:
        data = await self._request_json(
            f"{self._BASE_URL}/weather",
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": self._api_key,
                "units": "metric",
            },
        )
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        rain = data.get("rain", {})
        weather_list = data.get("weather", [])
        weather_entry = weather_list[0] if weather_list else {}

        return CurrentConditions(
            provider=self.provider,
            location_name=str(data.get("name", "")),
            latitude=latitude,
            longitude=longitude,
            temperature_celsius=float(main.get("temp", 0.0)),
            feels_like_celsius=float(main.get("feels_like", 0.0)),
            humidity_percent=float(main.get("humidity", 0.0)),
            pressure_hpa=float(main.get("pressure", 0.0)),
            wind_speed_mps=float(wind.get("speed", 0.0)),
            wind_direction_deg=wind.get("deg"),
            cloud_cover_percent=clouds.get("all"),
            precipitation_mm=rain.get("1h"),
            weather_code=str(weather_entry.get("id")) if weather_entry.get("id") is not None else None,
            description=str(weather_entry.get("description", "")),
            observed_at_epoch=float(data.get("dt", time.time())),
            raw=data,
        )

    async def get_forecast(self, latitude: float, longitude: float, hours: int) -> Forecast:
        data = await self._request_json(
            f"{self._BASE_URL}/forecast",
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": self._api_key,
                "units": "metric",
            },
        )
        entries = data.get("list", [])
        max_periods = max(1, hours // 3)
        periods: list[ForecastPeriod] = []

        for entry in entries[:max_periods]:
            main = entry.get("main", {})
            wind = entry.get("wind", {})
            pop = entry.get("pop")
            rain = entry.get("rain", {})
            weather_list = entry.get("weather", [])
            weather_entry = weather_list[0] if weather_list else {}
            start_epoch = float(entry.get("dt", 0.0))

            periods.append(
                ForecastPeriod(
                    start_epoch=start_epoch,
                    end_epoch=start_epoch + 3 * 3600,
                    temperature_celsius=float(main.get("temp", 0.0)),
                    temperature_min_celsius=main.get("temp_min"),
                    temperature_max_celsius=main.get("temp_max"),
                    humidity_percent=main.get("humidity"),
                    precipitation_probability_percent=float(pop) * 100.0 if pop is not None else None,
                    precipitation_mm=rain.get("3h"),
                    wind_speed_mps=wind.get("speed"),
                    weather_code=str(weather_entry.get("id")) if weather_entry.get("id") is not None else None,
                    description=str(weather_entry.get("description", "")),
                )
            )

        city = data.get("city", {})
        return Forecast(
            provider=self.provider,
            location_name=str(city.get("name", "")),
            latitude=latitude,
            longitude=longitude,
            periods=tuple(periods),
            fetched_at_epoch=time.time(),
            raw=data,
        )

    async def health_check(self) -> ProviderHealth:
        start = time.monotonic()
        try:
            await self._request_json(
                f"{self._BASE_URL}/weather",
                params={"lat": 0.0, "lon": 0.0, "appid": self._api_key, "units": "metric"},
            )
            latency = time.monotonic() - start
            status = HealthStatus.HEALTHY if latency < 3.0 else HealthStatus.DEGRADED
            return ProviderHealth(
                provider=self.provider,
                status=status,
                latency_seconds=latency,
                last_checked_epoch=time.time(),
            )
        except WeatherAuthError as exc:
            return ProviderHealth(
                provider=self.provider,
                status=HealthStatus.UNHEALTHY,
                latency_seconds=None,
                last_checked_epoch=time.time(),
                detail=str(exc),
            )
        except WeatherError as exc:
            return ProviderHealth(
                provider=self.provider,
                status=HealthStatus.DEGRADED,
                latency_seconds=time.monotonic() - start,
                last_checked_epoch=time.time(),
                detail=str(exc),
            )


class TomorrowIoClient(BaseWeatherClient):
    provider = WeatherProvider.TOMORROW_IO
    _BASE_URL = "https://api.tomorrow.io/v4"

    async def get_current(self, latitude: float, longitude: float) -> CurrentConditions:
        data = await self._request_json(
            f"{self._BASE_URL}/weather/realtime",
            params={
                "location": f"{latitude},{longitude}",
                "apikey": self._api_key,
                "units": "metric",
            },
        )
        values = data.get("data", {}).get("values", {})
        observed_time = data.get("data", {}).get("time")
        observed_epoch = self._parse_iso_to_epoch(observed_time) if observed_time else time.time()

        return CurrentConditions(
            provider=self.provider,
            location_name="",
            latitude=latitude,
            longitude=longitude,
            temperature_celsius=float(values.get("temperature", 0.0)),
            feels_like_celsius=float(values.get("temperatureApparent", values.get("temperature", 0.0))),
            humidity_percent=float(values.get("humidity", 0.0)),
            pressure_hpa=float(values.get("pressureSeaLevel", 0.0)),
            wind_speed_mps=float(values.get("windSpeed", 0.0)),
            wind_direction_deg=values.get("windDirection"),
            cloud_cover_percent=values.get("cloudCover"),
            precipitation_mm=values.get("precipitationIntensity"),
            weather_code=str(values.get("weatherCode")) if values.get("weatherCode") is not None else None,
            description=str(values.get("weatherCode", "")),
            observed_at_epoch=observed_epoch,
            raw=data,
        )

    async def get_forecast(self, latitude: float, longitude: float, hours: int) -> Forecast:
        data = await self._request_json(
            f"{self._BASE_URL}/weather/forecast",
            params={
                "location": f"{latitude},{longitude}",
                "apikey": self._api_key,
                "units": "metric",
                "timesteps": "1h",
            },
        )
        timelines = data.get("timelines", {})
        hourly = timelines.get("hourly", [])
        periods: list[ForecastPeriod] = []

        for entry in hourly[:hours]:
            values = entry.get("values", {})
            start_epoch = self._parse_iso_to_epoch(entry.get("time", ""))

            periods.append(
                ForecastPeriod(
                    start_epoch=start_epoch,
                    end_epoch=start_epoch + 3600,
                    temperature_celsius=float(values.get("temperature", 0.0)),
                    temperature_min_celsius=values.get("temperatureMin"),
                    temperature_max_celsius=values.get("temperatureMax"),
                    humidity_percent=values.get("humidity"),
                    precipitation_probability_percent=values.get("precipitationProbability"),
                    precipitation_mm=values.get("precipitationIntensity"),
                    wind_speed_mps=values.get("windSpeed"),
                    weather_code=str(values.get("weatherCode")) if values.get("weatherCode") is not None else None,
                    description=str(values.get("weatherCode", "")),
                )
            )

        return Forecast(
            provider=self.provider,
            location_name="",
            latitude=latitude,
            longitude=longitude,
            periods=tuple(periods),
            fetched_at_epoch=time.time(),
            raw=data,
        )

    async def health_check(self) -> ProviderHealth:
        start = time.monotonic()
        try:
            await self._request_json(
                f"{self._BASE_URL}/weather/realtime",
                params={"location": "0,0", "apikey": self._api_key, "units": "metric"},
            )
            latency = time.monotonic() - start
            status = HealthStatus.HEALTHY if latency < 3.0 else HealthStatus.DEGRADED
            return ProviderHealth(
                provider=self.provider,
                status=status,
                latency_seconds=latency,
                last_checked_epoch=time.time(),
            )
        except WeatherAuthError as exc:
            return ProviderHealth(
                provider=self.provider,
                status=HealthStatus.UNHEALTHY,
                latency_seconds=None,
                last_checked_epoch=time.time(),
                detail=str(exc),
            )
        except WeatherError as exc:
            return ProviderHealth(
                provider=self.provider,
                status=HealthStatus.DEGRADED,
                latency_seconds=time.monotonic() - start,
                last_checked_epoch=time.time(),
                detail=str(exc),
            )

    @staticmethod
    def _parse_iso_to_epoch(iso_string: str) -> float:
        try:
            import datetime

            normalized = iso_string.replace("Z", "+00:00")
            return datetime.datetime.fromisoformat(normalized).timestamp()
        except (ValueError, AttributeError):
            return time.time()


class CachedWeatherService:
    def __init__(
        self,
        clients: dict[WeatherProvider, WeatherClient],
        cache: Optional[TTLCache] = None,
        current_ttl_seconds: float = 300.0,
        forecast_ttl_seconds: float = 1800.0,
        fallback_order: Optional[list[WeatherProvider]] = None,
    ) -> None:
        self._clients = clients
        self._cache = cache or TTLCache()
        self._current_ttl_seconds = current_ttl_seconds
        self._forecast_ttl_seconds = forecast_ttl_seconds
        self._fallback_order = fallback_order or list(clients.keys())

    @staticmethod
    def _round_coord(value: float) -> float:
        return round(value, 3)

    def _current_cache_key(self, provider: WeatherProvider, latitude: float, longitude: float) -> str:
        return f"current:{provider.value}:{self._round_coord(latitude)}:{self._round_coord(longitude)}"

    def _forecast_cache_key(
        self, provider: WeatherProvider, latitude: float, longitude: float, hours: int
    ) -> str:
        return f"forecast:{provider.value}:{self._round_coord(latitude)}:{self._round_coord(longitude)}:{hours}"

    async def get_current(
        self, latitude: float, longitude: float, provider: Optional[WeatherProvider] = None
    ) -> CurrentConditions:
        providers = [provider] if provider else self._fallback_order
        last_error: Optional[Exception] = None

        for candidate in providers:
            client = self._clients.get(candidate)
            if client is None:
                continue

            cache_key = self._current_cache_key(candidate, latitude, longitude)
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

            try:
                result = await client.get_current(latitude, longitude)
                await self._cache.set(cache_key, result, self._current_ttl_seconds)
                return result
            except WeatherError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise WeatherError(WeatherProvider.OPENWEATHER, "no weather providers configured")

    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        hours: int = 24,
        provider: Optional[WeatherProvider] = None,
    ) -> Forecast:
        providers = [provider] if provider else self._fallback_order
        last_error: Optional[Exception] = None

        for candidate in providers:
            client = self._clients.get(candidate)
            if client is None:
                continue

            cache_key = self._forecast_cache_key(candidate, latitude, longitude, hours)
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

            try:
                result = await client.get_forecast(latitude, longitude, hours)
                await self._cache.set(cache_key, result, self._forecast_ttl_seconds)
                return result
            except WeatherError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise WeatherError(WeatherProvider.OPENWEATHER, "no weather providers configured")

    async def health_check_all(self) -> dict[WeatherProvider, ProviderHealth]:
        results = await asyncio.gather(
            *(client.health_check() for client in self._clients.values()),
            return_exceptions=True,
        )
        health_map: dict[WeatherProvider, ProviderHealth] = {}
        for provider_key, result in zip(self._clients.keys(), results):
            if isinstance(result, ProviderHealth):
                health_map[provider_key] = result
            else:
                health_map[provider_key] = ProviderHealth(
                    provider=provider_key,
                    status=HealthStatus.UNKNOWN,
                    latency_seconds=None,
                    last_checked_epoch=time.time(),
                    detail=str(result),
                )
        return health_map

    async def close(self) -> None:
        for client in self._clients.values():
            close_method = getattr(client, "close", None)
            if close_method is not None:
                await close_method()


def build_weather_service(
    openweather_api_key: Optional[str] = None,
    tomorrow_io_api_key: Optional[str] = None,
    current_ttl_seconds: float = 300.0,
    forecast_ttl_seconds: float = 1800.0,
    request_timeout_seconds: float = 10.0,
) -> CachedWeatherService:
    clients: dict[WeatherProvider, WeatherClient] = {}

    if openweather_api_key:
        clients[WeatherProvider.OPENWEATHER] = OpenWeatherClient(
            api_key=openweather_api_key,
            request_timeout_seconds=request_timeout_seconds,
        )
    if tomorrow_io_api_key:
        clients[WeatherProvider.TOMORROW_IO] = TomorrowIoClient(
            api_key=tomorrow_io_api_key,
            request_timeout_seconds=request_timeout_seconds,
        )

    return CachedWeatherService(
        clients=clients,
        current_ttl_seconds=current_ttl_seconds,
        forecast_ttl_seconds=forecast_ttl_seconds,
    )
