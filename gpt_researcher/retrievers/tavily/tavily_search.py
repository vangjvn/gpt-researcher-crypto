# Tavily API Retriever

# libraries
import os
from typing import Literal, Sequence, Optional
import requests
import json


from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Deque
from datetime import datetime, timedelta
from collections import deque
import threading
import os
from pydantic_settings import BaseSettings  # 更改这里
from pydantic import BaseModel, Field, ConfigDict



load_dotenv()


@dataclass
class APIKeyStatus:
    """API Key的状态信息"""
    key: str
    last_used: datetime = field(default_factory=datetime.now)
    request_count: int = 0
    is_available: bool = True
    error_count: int = 0
    last_error_time: Optional[datetime] = None

class RateLimit:
    """速率限制配置"""
    def __init__(self,
                 requests_per_second: float,
                 burst_limit: Optional[int] = None,
                 daily_limit: Optional[int] = None):
        self.requests_per_second = requests_per_second  # 每秒请求数
        self.min_interval = 1.0 / requests_per_second  # 最小请求间隔
        self.burst_limit = burst_limit  # 突发请求限制
        self.daily_limit = daily_limit  # 每日请求限制


class APIKeyPool:
    """API Key池管理器"""

    def __init__(self,
                 keys: str,
                 rate_limit: RateLimit):
        self.keys: Deque[APIKeyStatus] = deque(
            [APIKeyStatus(key=k) for k in keys.split(",") if k.strip()]
        )
        self.rate_limit = rate_limit
        self.lock = threading.Lock()
        self.last_reset = datetime.now()
        self._daily_count = 0

    def get_next_key(self) -> Optional[str]:
        """获取下一个可用的API Key"""
        with self.lock:
            if not self.keys:
                return None

            current_time = datetime.now()

            # 重置每日计数
            if (current_time - self.last_reset).days >= 1:
                self._daily_count = 0
                self.last_reset = current_time

            # 检查每日限制
            if (self.rate_limit.daily_limit and
                    self._daily_count >= self.rate_limit.daily_limit):
                return None

            # 尝试找到一个可用的key
            attempts = len(self.keys)
            while attempts > 0:
                key_status = self.keys[0]

                # 检查这个key是否可用
                time_since_last_use = (current_time - key_status.last_used).total_seconds()
                if (time_since_last_use >= self.rate_limit.min_interval and
                        key_status.is_available):
                    key_status.last_used = current_time
                    key_status.request_count += 1
                    self._daily_count += 1

                    # 将使用过的key放到队列末尾
                    self.keys.rotate(-1)
                    return key_status.key

                # 如果当前key不可用，轮转到下一个
                self.keys.rotate(-1)
                attempts -= 1

            return None


class APIKeyManager:
    """API Key管理器"""

    def __init__(self):
        self.pools: Dict[str, APIKeyPool] = {}

    def initialize_pool(self,
                        pool_name: str,
                        keys: str,
                        rate_limit: RateLimit):
        """初始化特定的API Key池"""
        self.pools[pool_name] = APIKeyPool(keys, rate_limit)

    def get_next_key(self, pool_name: str) -> Optional[str]:
        """从指定的池中获取下一个可用的key"""
        if pool_name in self.pools:
            return self.pools[pool_name].get_next_key()
        return None

class Settings(BaseSettings):
    # API Key管理器
    api_key_manager: APIKeyManager = Field(default_factory=APIKeyManager)

    TAVILY_API_KEYS: str = os.getenv("TAVILY_API_KEYS","")


    def __init__(self, **data: Any):
        super().__init__(**data)
        self._initialize_api_key_pools()

    def _initialize_api_key_pools(self):
        """初始化所有API Key池"""
        # 从环境变量读取多个API key

        print("开始初始化api key池")
        self.api_key_manager.initialize_pool(
            "tavily",
            self.TAVILY_API_KEYS,
            RateLimit(
                requests_per_second=10.0,  # OpenAI的限制是每秒80次
                daily_limit=1000  # 示例每日限制
            )
        )

    def get_tavily_api_key(self) -> Optional[str]:
        """获取下一个可用的tavily API Key"""
        return self.api_key_manager.get_next_key("tavily")

settings = Settings()



class TavilySearch():
    """
    Tavily API Retriever
    """

    def __init__(self, query, headers=None, topic="general"):
        """
        Initializes the TavilySearch object
        Args:
            query:
        """
        self.query = query
        self.headers = headers or {}
        self.topic = topic
        self.base_url = "https://api.tavily.com/search"
        self.api_key = self.get_api_key()
        self.headers = {
            "Content-Type": "application/json",
        }

    def get_api_key(self):
        """
        Gets the Tavily API key
        Returns:

        """
        api_key = self.headers.get("tavily_api_key")
        if not api_key:
            try:
                api_key = settings.get_tavily_api_key()

            except KeyError:
                raise Exception(
                    "Tavily API key not found. Please set the TAVILY_API_KEY environment variable.")
        return api_key

    def _search(self,
                query: str,
                search_depth: Literal["basic", "advanced"] = "basic",
                topic: str = "general",
                days: int = 2,
                max_results: int = 5,
                include_domains: Sequence[str] = None,
                exclude_domains: Sequence[str] = None,
                include_answer: bool = False,
                include_raw_content: bool = False,
                include_images: bool = False,
                use_cache: bool = True,
                ) -> dict:
        """
        Internal search method to send the request to the API.
        """

        data = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "days": days,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_images": include_images,
            "api_key": self.api_key,
            "use_cache": use_cache,
        }

        response = requests.post(self.base_url, data=json.dumps(
            data), headers=self.headers, timeout=100)

        if response.status_code == 200:
            return response.json()
        else:
            # Raises a HTTPError if the HTTP request returned an unsuccessful status code
            response.raise_for_status()

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        try:
            # Search the query
            results = self._search(
                self.query, search_depth="basic", max_results=max_results, topic=self.topic)
            sources = results.get("results", [])
            if not sources:
                raise Exception("No results found with Tavily API search.")
            # Return the results
            search_response = [{"href": obj["url"],
                                "body": obj["content"]} for obj in sources]
        except Exception as e:
            print(
                f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
