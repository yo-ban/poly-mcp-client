# -*- coding: utf-8 -*-
"""
poly-mcp-client

An MCP client adapting tool definitions for Anthropic, OpenAI, and Gemini.
"""

# ライブラリのバージョン
__version__ = "0.0.1" # 例: セマンティックバージョニングに従う

# 主要なクラスをインポートして公開
from .manager import PolyMCPClient

# 設定に必要なモデルクラスもインポートして公開 (利用者が型ヒントや設定作成に使えるように)
from .models import (
    StdioServerConfig,
    HttpServerConfig,
    ServerConfig, # Union型も公開しておくと便利
    McpServersConfig # ルートの設定モデル
    # 必要に応じて他の公開したいモデルも追加
)

# (オプション) ライブラリ固有の例外を公開する場合
# from .exceptions import MCPManagerError, MCPConnectionError

# __all__ を定義して、from mcp_client_manager import * でインポートされるものを明示的に指定
__all__ = [
    "PolyMCPClient",
    "StdioServerConfig",
    "HttpServerConfig",
    "ServerConfig",
    "McpServersConfig",
    "__version__",
    # 公開する例外クラスもここに追加
]

# (オプション) パッケージレベルのロガー設定 (ハンドラは設定しない)
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
