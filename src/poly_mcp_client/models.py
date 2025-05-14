from pydantic import BaseModel, Field, ValidationError, RootModel
from typing import Dict, List, Any, Optional, Literal, Union, Annotated, TypedDict


class StdioServerConfig(BaseModel):
    """Stdioサーバー固有の設定"""
    type: Literal["stdio"] = "stdio"  # デフォルトと型を同時に指定
    command: str
    args: List[str] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = None


class HttpServerConfig(BaseModel):
    """HTTPサーバー固有の設定"""
    type: Literal["http"] = "http"  # デフォルトと型を同時に指定
    url: str
    # 必要に応じて認証情報などを追加


class StreamableHttpServerConfig(BaseModel):
    """Streamable HTTPサーバー固有の設定"""
    type: Literal["streamable-http"] = "streamable-http"
    url: str
    # 必要に応じて認証情報などを追加


# Union型で Stdio または Http 設定を受け入れる
ServerConfig = Annotated[Union[StdioServerConfig,HttpServerConfig,StreamableHttpServerConfig], Field(discriminator="type")]

class McpServersConfig(RootModel[Dict[str, ServerConfig]]):
    root: Dict[str, ServerConfig]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


# --- 内部で使用するサーバー定義  ---
class InternalServerDefinition(BaseModel):
    name: str
    type: str  # "stdio", "http", or "streamable-http"
    config: ServerConfig

# --- カノニカル形式のツールパラメータと定義  ---
class CanonicalToolItemsSchema(TypedDict):
    # 配列要素の型 ('string', 'integer', 'number', 'boolean', 'object', 'array')
    type: str


class CanonicalToolParameter(TypedDict):
    type: str  # パラメータの型
    description: Optional[str]
    items: Optional[CanonicalToolItemsSchema]  # typeが'array'の場合に使用


class CanonicalToolDefinition(TypedDict):
    name: str
    description: Optional[str]
    parameters: Dict[str, CanonicalToolParameter]  # プロパティ名 -> パラメータ定義
    required: List[str]