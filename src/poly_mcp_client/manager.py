
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional, Literal, Tuple
import logging
from pydantic import ValidationError

# MCP SDKのインポート
from mcp import ClientSession, StdioServerParameters, McpError
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import mcp.types as types
import httpx

from .constants import MCP_PREFIX
from .models import (
    StdioServerConfig, 
    HttpServerConfig, 
    McpServersConfig, 
    InternalServerDefinition, 
    CanonicalToolParameter, 
    CanonicalToolDefinition
)

# --- ロギング設定 ---
# ライブラリ利用者が設定することを想定
logger = logging.getLogger(__name__)

class PolyMCPClient:
    def __init__(self, mcp_prefix: str = MCP_PREFIX):
        self._sessions: Dict[str, ClientSession] = {}
        self._server_capabilities: Dict[str,Optional[types.ServerCapabilities]] = {}
        self._server_definitions: Dict[str, InternalServerDefinition] = {}
        self._connection_tasks: Dict[str, asyncio.Task] = {}
        self._initial_connection_futures: Dict[str, asyncio.Future] = {}
        self._is_initialized = False
        self._lock = asyncio.Lock()
        self._is_shutting_down = False
        self._mcp_prefix = mcp_prefix

    async def initialize(self, config_path: Optional[str] = None, config_data: Optional[Dict[str, Any]] = None):
        """
        指定された設定ファイルまたはデータからMCPサーバー接続を初期化する。
        config_path が指定された場合はファイルを読み込む。
        config_data が指定された場合は、{"mcpServers": {"server_key": {...}, ...}} の形式を期待する。
        """
        async with self._lock:
            if self._is_initialized:
                logger.warning("MCPClientManager is already initialized.")
                return
            if self._is_shutting_down:
                logger.warning("MCPClientManager is shutting down. Cannot initialize.")
                return

            logger.info("MCPClientManager is initializing...")
            servers_dict_data: Optional[Dict] = None
            if config_path:
                logger.info(f"Loading configuration file {config_path}...")
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        raw_config_data = json.load(f)
                    # ファイルから読み込んだ場合、mcpServers キーをチェック
                    if "mcpServers" not in raw_config_data or not isinstance(raw_config_data.get("mcpServers"), dict):
                        logger.error(f"Invalid configuration file {config_path}. Top level must contain 'mcpServers' object.")
                        return
                    servers_dict_data = raw_config_data["mcpServers"] # 辞書の中身を取得

                except FileNotFoundError:
                    logger.error(f"Configuration file not found: {config_path}")
                    return
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in configuration file: {e}")
                    return

            elif config_data:
                logger.info("Using provided configuration data.")

                if isinstance(config_data, dict) and "mcpServers" in config_data and isinstance(config_data.get("mcpServers"), dict):
                    servers_dict_data = config_data["mcpServers"]
                else:
                    logger.error("Invalid configuration data. Top level must contain 'mcpServers' object.")
                    return
            else:
                logger.warning("No configuration file path or data provided. Skipping initialization.")
                return
            
            if not servers_dict_data:
                logger.info("No server configurations provided. No servers will be connected.")
                self._is_initialized = True  # 空でも初期化は完了
                return

            # --- Pydantic 検証 ---
            self._server_definitions.clear()
            self._initial_connection_futures.clear()
            validated_count = 0
            try:
                # McpServersConfig を使用して辞書全体を検証
                parsed_servers = McpServersConfig.model_validate(servers_dict_data)

                for server_name, server_config in parsed_servers.root.items():
                    # ServerConfig は Union なので、type で分岐する必要はない
                    internal_def = InternalServerDefinition(
                        name=server_name,
                        type=server_config.type,  # type は StdioServerConfig/HttpServerConfig に含まれる
                        config=server_config
                    )
                    self._server_definitions[server_name] = internal_def
                    self._initial_connection_futures[server_name] = asyncio.Future()
                    validated_count += 1
                logger.info(f"{validated_count} MCP server configurations validated successfully.")

            except ValidationError as e:
                logger.error(f"MCP server configuration validation error:\n{e}")
                # エラー発生時、Futureをキャンセルまたはエラー状態にする
                for name in self._server_definitions:
                    if name in self._initial_connection_futures and not self._initial_connection_futures[name].done():
                        self._initial_connection_futures[name].set_exception(e)
                self._server_definitions.clear() # エラーがあったら定義もクリア
                return

            # 接続タスク開始
            for name, definition in self._server_definitions.items():
                # Future を接続タスクに渡す
                self._start_connection(name, definition, self._initial_connection_futures[name])

            self._is_initialized = True
            logger.info("MCPClientManager initialization complete. Connections will be attempted in the background.")
            logger.info("To wait for initial connections, call await manager.wait_for_initial_connections()")

    async def wait_for_initial_connections(self, timeout: Optional[float] = None) -> Dict[str, Tuple[bool, Optional[Exception]]]:
        """
        initializeで開始された全てのサーバーへの初期接続試行が完了するまで待機する。

        Args:
            timeout: 全体の待機タイムアウト時間 (秒)。Noneの場合は無制限。

        Returns:
            サーバー名をキーとし、(接続成功フラグ, 例外) のタプルを値とする辞書。
            接続成功フラグが False の場合、例外オブジェクトが含まれる。
        """
        if not self._is_initialized and not self._server_definitions:
            logger.warning("Manager is not initialized or no servers are configured.")
            return {}

        futures = list(self._initial_connection_futures.values())
        if not futures:
            logger.info("No initial connection futures to wait for.")
            return {}

        logger.info(f"{len(futures)} servers to wait for initial connection (timeout: {timeout}s)...")

        done, pending = await asyncio.wait(futures, timeout=timeout, return_when=asyncio.ALL_COMPLETED)

        results: Dict[str, Tuple[bool, Optional[Exception]]] = {}
        server_name_map = {f: name for name, f in self._initial_connection_futures.items()}

        for fut in done:
            server_name = server_name_map.get(fut, "Unknown")
            try:
                result = fut.result() # 結果を取得 (成功ならTrue、失敗なら例外がraiseされる)
                results[server_name] = (True, None)
                logger.info(f"Server '{server_name}' initial connection successful.")
            except Exception as e:
                results[server_name] = (False, e)
                logger.error(f"Server '{server_name}' initial connection error: {e}", exc_info=False)

        if pending:
            logger.warning(f"Timeout ({timeout}s) reached. Some initial connections did not complete: {len(pending)}")
            for fut in pending:
                server_name = server_name_map.get(fut, "Unknown")
                # Futureをキャンセルしてリソースリークを防ぐ試み
                fut.cancel()
                results[server_name] = (False, asyncio.TimeoutError(f"Initial connection timed out after {timeout}s"))

        logger.info("Initial connection wait complete.")
        return results

    def _start_connection(self, name: str, definition: InternalServerDefinition, initial_conn_future: asyncio.Future) -> Optional[asyncio.Task]:
        """個別のサーバーへの接続タスクを作成し、管理する"""
        
        if self._is_shutting_down:
            logger.warning(f"Shutdown in progress. Connection task for server '{name}' will not be started.")
            if not initial_conn_future.done():
                initial_conn_future.set_exception(RuntimeError("Shutdown in progress"))
            return None

        if name in self._connection_tasks and not self._connection_tasks[name].done():
            logger.info(f"Connection task for server '{name}' is already running.")
            # 既に実行中ならFutureは完了しているはずだが、念のためチェック
            if not initial_conn_future.done():
                # 既存タスクが完了するのを待つか、即時完了とするか？
                # ここではシンプルに、既にタスクがあれば接続試行は完了しているとみなす
                # （より厳密には既存タスクの結果をFutureに設定する必要があるかもしれない）
                initial_conn_future.set_result(True) # 仮に成功扱い
            return self._connection_tasks[name]

        logger.info(f"Starting connection and monitoring task for server '{name}' ({definition.type})...")
        task = asyncio.create_task(self._connect_and_monitor(name, definition, initial_conn_future))
        self._connection_tasks[name] = task
        # タスク完了時に辞書から削除するコールバック
        task.add_done_callback(
            lambda t, n=name: self._connection_tasks.pop(n, None)
        )
        return task

    async def _connect_and_monitor(self, name: str, definition: InternalServerDefinition, initial_conn_future: asyncio.Future):
        """サーバーへの接続、セッション確立、プロセス監視を行うコルーチン (Futureを引数に追加)"""
        retry_delay = 5
        max_retry_delay = 60
        initial_attempt = True # 最初の接続試行かどうかを追跡

        while not self._is_shutting_down:  # 初期化フラグではなくシャットダウンフラグを見る
            session = None
            exit_stack = AsyncExitStack()
            capabilities = None  # ループ内でリセット
            connection_error = None # 初期接続エラー保持用

            try:
                if definition.type == "stdio":
                    if not isinstance(definition.config, StdioServerConfig):
                        logger.error(f"Invalid config type for stdio server: {name}")
                        connection_error = TypeError("Invalid config type for stdio server.")
                        break

                    stdio_config = definition.config
                    server_params = StdioServerParameters(
                        command=stdio_config.command,
                        args=stdio_config.args,
                        env=stdio_config.env
                    )
                    logger.info(f"Attempting connection to server '{name}': {stdio_config.command} {' '.join(stdio_config.args)}")

                    transport_streams = await exit_stack.enter_async_context(stdio_client(server_params))
                    reader, writer = transport_streams
                    logger.info(f"Transport connection to server '{name}' complete.")

                    session = await exit_stack.enter_async_context(ClientSession(reader, writer))
                    # セッション初期化
                    init_result = await asyncio.wait_for(session.initialize(), timeout=30.0)

                    self._sessions[name] = session
                    capabilities = init_result.capabilities
                    self._server_capabilities[name] = capabilities

                    logger.info(f"MCP connection to server '{name}' established.")
                    if capabilities:
                        logger.info(f"Server '{name}' capabilities: {capabilities}")

                    retry_delay = 5  # 接続成功でリトライ遅延をリセット

                    # --- 接続維持ループ (stdio) ---
                    # 最初の接続試行が成功したらFutureを完了
                    if initial_attempt and not initial_conn_future.done():
                        initial_conn_future.set_result(True)
                    initial_attempt = False

                    # 接続が維持されているか、またはシャットダウンされるまで待機
                    while not self._is_shutting_down and name in self._sessions:
                        # TODO: ClientSession の接続状態を確認する方法があれば追加
                        await asyncio.sleep(5)
                    
                    if self._is_shutting_down:
                        logger.info(f"Shutdown requested. Stopping monitoring of server '{name}' (stdio).")
                        break
                    logger.warning(f"Connection to server '{name}' (stdio) lost.")
                    # 接続が失われた場合はループの最後でリトライされる

                elif definition.type == "http":
                    if not isinstance(definition.config, HttpServerConfig):
                        logger.error(
                            f"Invalid config type for http server: {name}")
                        connection_error = TypeError("Invalid config type for http server.")
                        break
                    http_config = definition.config
                    server_url = http_config.url
                    logger.info(
                        f"Attempting connection to server '{name}' (http): {server_url}")

                    try:
                        # sse_client コンテキストマネージャを使用
                        # 認証ヘッダー等が必要な場合、sse_clientが引数で受け付けるか確認が必要
                        transport_streams = await exit_stack.enter_async_context(sse_client(server_url))
                        reader, writer = transport_streams
                        logger.info(f"Transport connection to server '{name}' (http) complete.")

                        session = await exit_stack.enter_async_context(ClientSession(reader, writer))
                        init_result = await asyncio.wait_for(session.initialize(), timeout=30.0)

                        self._sessions[name] = session
                        capabilities = init_result.capabilities
                        self._server_capabilities[name] = capabilities
                        logger.info(f"MCP connection to server '{name}' (http) established.")
                        if capabilities:
                            logger.info(f"Server '{name}' (http) capabilities: {capabilities}")

                        retry_delay = 5

                    # HTTP/SSE 接続時のエラーハンドリング
                    except httpx.RequestError as e:
                        logger.error(f"HTTP request error during connection to server '{name}' (http): {e}")
                        connection_error = e
                        # リトライのためにループの最後へ
                    except httpx.HTTPStatusError as e:
                        logger.error(f"Server '{name}' (http) returned error status: {e.response.status_code} {e.response.reason_phrase}")
                        connection_error = e
                        # リトライのためにループの最後へ
                    except ConnectionRefusedError as e: # sse_client 内で発生する可能性
                        logger.error(f"Connection to server '{name}' (http) refused. Check URL: {server_url}")
                        connection_error = e
                    except asyncio.TimeoutError as e: # initialize のタイムアウト
                        logger.error(f"Server '{name}' (http) initialization timed out.")
                        connection_error = e
                    except Exception as e: # その他の予期せぬ接続エラー
                        logger.error(f"Unexpected connection error during connection to server '{name}' (http): {e}", exc_info=False)
                        connection_error = e
                        break # 回復不能かもしれないのでループを抜ける

                    # 接続成功時の処理
                    if session and not connection_error:
                        if initial_attempt and not initial_conn_future.done():
                            initial_conn_future.set_result(True)
                        initial_attempt = False

                        # --- 接続維持ループ (http/sse) ---
                        while not self._is_shutting_down and name in self._sessions:
                            # TODO: SSE接続の状態を確認する方法があれば追加
                            # sse_client や ClientSession が切断を検知する仕組みがあるか？
                            # なければ定期的なpingなどで確認する必要があるかもしれない
                            await asyncio.sleep(5)

                        if self._is_shutting_down:
                            logger.info(f"Shutdown requested. Stopping monitoring of server '{name}' (http).")
                            break
                        logger.warning(f"Connection to server '{name}' (http) lost.")
                        # 接続が失われた場合はループの最後でリトライされる
                    else:
                        # 接続に失敗した場合 (connection_error が設定されているはず)
                        # ループの最後でリトライされる
                        pass
                else:
                    connection_error = ValueError(f"Unknown server type: {definition.type} ({name})")
                    logger.error(connection_error)
                    break

            # --- 接続失敗時の処理 ---
            except (ConnectionRefusedError, FileNotFoundError, asyncio.TimeoutError) as e:
                connection_error = e # エラーを記録
                logger.error(f"Failed to connect to server '{name}': {e}", exc_info=False)
                # initial_attempt フラグは finally で False にする
            except asyncio.CancelledError:
                logger.info(f"Connection task for server '{name}' was cancelled.")
                connection_error = asyncio.CancelledError() # キャンセルも記録
                break # キャンセルされたらループ終了
            except Exception as e:
                connection_error = e # その他の予期せぬエラー
                logger.error(f"Unexpected error occurred during connection or monitoring of server '{name}': {e}", exc_info=False)
                # initial_attempt フラグは finally で False にする
            finally:
                # 最初の試行でエラーが発生した場合、Futureを完了させる
                if initial_attempt and not initial_conn_future.done():
                    if connection_error:
                        initial_conn_future.set_exception(connection_error)
                    else:
                        # ここに来ることは稀だが、念のため
                        initial_conn_future.set_exception(RuntimeError(f"Unknown initial connection state for server '{name}'."))
                initial_attempt = False # 最初の試行はこれで完了

                # セッションとケイパビリティ情報を削除
                if name in self._sessions:
                    del self._sessions[name]
                    logger.debug(f"Deleted session info for server '{name}'.")
                if name in self._server_capabilities:
                    del self._server_capabilities[name]
                    logger.debug(f"Deleted capability info for server '{name}'.")

                # exit_stackでリソースをクリーンアップ
                try:
                    await exit_stack.aclose()
                    logger.debug(f"Cleaned up resource stack for server '{name}'.")
                except Exception as e_stack:
                    logger.error(f"Error cleaning up resource stack for server '{name}': {e_stack}")

                if self._is_shutting_down:
                    logger.info(f"Shutdown in progress. Reconnection for server '{name}' will not be attempted.")
                    break # シャットダウン中はループ終了

            # 再接続ロジック (キャンセルされていない場合)
            # リトライすべきでないエラータイプの場合はループを抜ける
            if connection_error and isinstance(connection_error, (TypeError, ValueError)):
                logger.error(f"Server '{name}' ({definition.type}) is not recoverable due to configuration error. Error: {connection_error}")
                break
            if connection_error and isinstance(connection_error, NotImplementedError): # HTTP未実装の場合など
                logger.error(f"Server '{name}' ({definition.type}) is not implemented or supported. Reconnection will not be attempted.")
                break

            logger.info(f"Attempting reconnection to server '{name}' in {retry_delay} seconds...")
            try:
                
                await asyncio.sleep(retry_delay)  # キャンセル可能にする
            except asyncio.CancelledError:
                logger.info(f"Reconnection wait cancelled ({name}).")
                break  # キャンセルされたらループ終了
            retry_delay = min(retry_delay * 2, max_retry_delay)

        logger.info(f"Connection and monitoring task for server '{name}' ({definition.type}) ended.")

        # タスク終了時に Future がまだ完了していなければ、エラー状態にする
        if not initial_conn_future.done():
            final_error = connection_error or RuntimeError(f"Connection task for '{name}' ({definition.type}) ended unexpectedly.")
            if isinstance(final_error, asyncio.CancelledError): # キャンセルで終了した場合
                initial_conn_future.cancel()
            else:
                initial_conn_future.set_exception(final_error)

    def _mcp_tool_to_canonical(self, server_name: str, mcp_tool: types.Tool) -> CanonicalToolDefinition:
        """MCP Toolオブジェクトをカノニカル形式の辞書に変換する"""
        parameters: Dict[str, CanonicalToolParameter] = {}
        required: List[str] = []

        # inputSchemaの処理ロジック
        if mcp_tool.inputSchema and isinstance(mcp_tool.inputSchema, dict):
            input_schema_dict = mcp_tool.inputSchema
            schema_type = input_schema_dict.get("type", "object")
            properties = input_schema_dict.get("properties")

            if schema_type == "object" and properties and isinstance(properties, dict):
                required = input_schema_dict.get("required", [])
                if not isinstance(required, list):
                    logger.warning(f"Tool '{server_name}/{mcp_tool.name}' required is not a list: {required}")
                    required = []

                for param_name, schema_prop in properties.items():
                    if not isinstance(schema_prop, dict):
                        logger.warning(f"Tool '{server_name}/{mcp_tool.name}' parameter '{param_name}' schema definition is not a dictionary: {type(schema_prop)}")
                        continue

                    param_type = schema_prop.get("type", "any")
                    param_data: CanonicalToolParameter = {
                        "type": param_type,
                    }

                    # description が None でない場合のみ辞書に追加
                    param_description = schema_prop.get("description")
                    if param_description is not None:
                        param_data["description"] = param_description

                    if param_type == "array":
                        items_schema = schema_prop.get("items")
                        if isinstance(items_schema, dict):
                            item_type = items_schema.get("type", "any")
                            param_data["items"] = {"type": item_type}
                        elif items_schema is not None:
                            logger.warning(f"Tool '{server_name}/{mcp_tool.name}' array parameter '{param_name}' items schema is not a dictionary: {type(items_schema)}")

                    parameters[param_name] = param_data
            elif properties and isinstance(properties, dict) and not input_schema_dict.get("type"):
                logger.debug(f"Tool '{server_name}/{mcp_tool.name}' inputSchema is type='object' but not specified. Processing properties.")
                required = input_schema_dict.get("required", [])
                if not isinstance(required, list):
                    logger.warning(f"Tool '{server_name}/{mcp_tool.name}' required is not a list: {required}")
                    required = []
                for param_name, schema_prop in properties.items():
                    if not isinstance(schema_prop, dict):
                        logger.warning(f"Tool '{server_name}/{mcp_tool.name}' parameter '{param_name}' schema definition is not a dictionary: {type(schema_prop)}")
                        continue

                    param_type = schema_prop.get("type", "any")
                    param_data: CanonicalToolParameter = {
                        "type": param_type,
                    }

                    param_description = schema_prop.get("description")
                    if param_description is not None:
                        param_data["description"] = param_description

                    if param_type == "array":
                        items_schema = schema_prop.get("items")
                        if isinstance(items_schema, dict):
                            item_type = items_schema.get("type", "any")
                            param_data["items"] = {"type": item_type}
                        elif items_schema is not None:
                            logger.warning(f"Tool '{server_name}/{mcp_tool.name}' array parameter '{param_name}' items schema is not a dictionary: {type(items_schema)}")
                    parameters[param_name] = param_data

        elif mcp_tool.inputSchema:
            logger.warning(f"Tool '{server_name}/{mcp_tool.name}' inputSchema is not a dictionary: type={type(mcp_tool.inputSchema)}")

        # ツール名に 'mcp_' プレフィックスとサーバー名を付与
        canonical_name = f"{self._mcp_prefix}{server_name}-{mcp_tool.name}"
        canonical_def: CanonicalToolDefinition = {
            "name": canonical_name,
            "parameters": parameters,
            "required": required
        }
        # description が None でない場合のみ辞書に追加
        if mcp_tool.description is not None:
            canonical_def["description"] = mcp_tool.description
        return canonical_def
    
    def _convert_tool_for_vendor(self, tool_def: CanonicalToolDefinition, vendor: Literal["anthropic", "openai", "google"]) -> Dict[str, Any]:
        """CanonicalToolDefinition を指定されたベンダー形式の辞書に変換する"""
        if vendor == "openai":
            # OpenAI format
            return {
                "type": "function",
                "function": {
                    "name": tool_def["name"],
                    "description": tool_def["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool_def["parameters"],
                        "required": tool_def["required"],
                        # "additionalProperties": False # OpenAI では通常指定しないことが多い
                    },
                    # "strict": False # 通常指定しない
                }
            }
        elif vendor == "anthropic":
            # Anthropic format
            return {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool_def["parameters"],
                    "required": tool_def["required"]
                }
            }
        elif vendor == "google":
            # Gemini format (Dictionary representation)
            properties = {}
            for param_name, param_def in tool_def["parameters"].items():
                gemini_type = "STRING"  # Default
                items_schema = None

                if param_def["type"] == "string":
                    gemini_type = "STRING"
                elif param_def["type"] == "integer":
                    gemini_type = "NUMBER"
                elif param_def["type"] == "number":
                    gemini_type = "NUMBER"
                elif param_def["type"] == "boolean":
                    gemini_type = "BOOLEAN"
                elif param_def["type"] == "array":
                    gemini_type = "ARRAY"
                    # Default item type is STRING if not specified
                    item_type_str = param_def.get("items", {}).get("type", "string")
                    item_gemini_type = "STRING"
                    if item_type_str == "integer":
                        item_gemini_type = "NUMBER"
                    elif item_type_str == "number":
                        item_gemini_type = "NUMBER"
                    elif item_type_str == "boolean":
                        item_gemini_type = "BOOLEAN"
                    elif item_type_str == "object":
                        # Array of objects might need further definition if nested schema is complex
                        item_gemini_type = "OBJECT"

                    items_schema = {"type": item_gemini_type}
                elif param_def["type"] == "object":
                    gemini_type = "OBJECT"
                    # Note: Nested object properties are not explicitly handled here.
                    # Gemini might require a more detailed schema for nested objects.
                elif param_def["type"] == "any":
                    # Gemini doesn't have a direct 'any' type. Defaulting to STRING or OBJECT might be options.
                    # Or omit the type if possible/allowed by Gemini API. Let's default to STRING.
                    gemini_type = "STRING"
                    logger.warning(f"Gemini conversion: parameter '{param_name}' type 'any' is handled as 'STRING'.")

                prop_entry = {
                    "type": gemini_type,
                    "description": param_def.get("description")
                }
                if items_schema:
                    prop_entry["items"] = items_schema

                properties[param_name] = prop_entry

            # Gemini FunctionDeclaration structure as a dictionary
            return {
                "name": tool_def["name"],
                "description": tool_def.get("description"),
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": tool_def.get("required", [])
                }
            }
        else:
            # Should not happen if using Literal for vendor type hint
            raise ValueError(f"Unsupported vendor: {vendor}")
    
    # ツール定義の取得
    async def get_available_tools(
        self,
        vendor: Optional[Literal["anthropic", "openai", "google", "canonical"]] = "anthropic"
    ) -> List[Dict[str, Any]]:
        """
        現在接続中の全ての有効なMCPサーバーから利用可能なツールリストを取得する。

        Args:
            vendor: 指定された場合、ツール定義を指定ベンダーの形式に変換する。
                    Noneの場合、カノニカル形式で返す。

        Returns:
            List[Dict[str, Any]]: ツール定義のリスト。形式は vendor 引数に依存する。Googleの場合も辞書形式のリスト。
        """
        if not self._is_initialized:
            logger.warning("MCPClientManager is not initialized. Returning empty tool list.")
            return []
        if self._is_shutting_down:
            logger.warning("MCPClientManager is shutting down. Returning empty tool list.")
            return []

        all_canonical_tools: List[CanonicalToolDefinition] = []
        tasks = []
        # Read active sessions within the lock to avoid race conditions during shutdown
        async with self._lock:
            active_sessions_data = list(self._sessions.items())  # イテレーション用コピー

        async def fetch_and_convert_tools(name: str, session: ClientSession):
            # 保存しておいたケイパビリティ情報を参照
            # Use get to avoid KeyError if capability info hasn't arrived yet or was cleared
            capabilities = self._server_capabilities.get(name)
            if capabilities and capabilities.tools:
                try:
                    # ツールリスト取得にタイムアウトを追加
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=30.0)
                    # session.list_tools() の結果が None でないことを確認
                    if tools_result:
                        for mcp_tool in tools_result.tools:
                            # ツールがNoneでないことも確認
                            if mcp_tool:
                                try:
                                    canonical_tool = self._mcp_tool_to_canonical(name, mcp_tool)
                                    all_canonical_tools.append(canonical_tool)
                                except Exception as conversion_err:
                                    tool_name_str = getattr(mcp_tool, 'name', 'N/A')
                                    logger.error(f"Server '{name}' tool '{tool_name_str}' canonical conversion failed: {conversion_err}", exc_info=False)
                            else:
                                logger.warning(f"Server '{name}' returned None tool object. Skipping.")
                    else:
                        logger.warning(f"Server '{name}' returned None tool list.")

                except asyncio.TimeoutError:
                    logger.error(f"Server '{name}' tool list retrieval timed out.")
                except Exception as e:
                    # 接続が既に切れている場合なども考慮
                    logger.error(f"Server '{name}' tool list retrieval or conversion error: {e}", exc_info=False)
            else:
                logger.info(f"Server '{name}' does not support tool functionality or does not have capability information.")

        for name, session in active_sessions_data:
            # セッションがNoneでないことを確認 (念のため)
            if session:
                tasks.append(fetch_and_convert_tools(name, session))

        if tasks:
            # gather は例外を集約しないため、fetch_and_convert_tools 内でエラー処理が必要
            await asyncio.gather(*tasks)

        logger.info(f"{len(all_canonical_tools)} canonical MCP tool definitions obtained.")

        # ベンダー形式への変換
        if vendor != "canonical":
            vendor_tools = []
            for canonical_tool in all_canonical_tools:
                try:
                    vendor_tool = self._convert_tool_for_vendor(canonical_tool, vendor)
                    vendor_tools.append(vendor_tool)
                except Exception as e:
                    logger.error(f"Tool '{canonical_tool.get('name', 'N/A')}' conversion to {vendor} format failed: {e}", exc_info=False)
            logger.info(f"{len(vendor_tools)} tool definitions converted to {vendor} format.")
            return vendor_tools
        else:
            # vendor指定なしの場合はカノニカル形式のリストを返す
            # CanonicalToolDefinition は TypedDict なので、そのまま辞書リストとして返せる
            return all_canonical_tools
    

    async def execute_mcp_tool(self, full_tool_name: str, arguments: Dict[str, Any]) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """
        指定されたMCPサーバーのツールを実行する。
        full_tool_name は "mcp_server_name_tool_name" の形式。
        サーバー名やツール名にはハイフンが含まれる可能性がある。
        """
        if not self._is_initialized:
            raise RuntimeError("MCPClientManager is not initialized.")
        if self._is_shutting_down:
            raise RuntimeError("MCPClientManager is shutting down. Cannot execute tools.")

        if not full_tool_name.startswith(self._mcp_prefix):
            raise ValueError(f"Tool name '{full_tool_name}' must be in the format '{self._mcp_prefix}server_name-tool_name'.")

        # 'プレフィックスを除去
        name_suffix = full_tool_name[len(self._mcp_prefix):]

        server_name: Optional[str] = None
        original_tool_name: Optional[str] = None

        # 登録されているサーバー名で前方一致検索
        # 例: mcp-my-server-1-get-tool-data -> my-server-1 が server_name, get-tool-data が tool_name
        # 例: mcp-files-list-files -> files が server_name, list-files が tool_name
        found_server_key = None
        for server_key in self._server_definitions.keys():
            prefix_to_check = f"{server_key}-"
            if name_suffix.startswith(prefix_to_check):
                # 最も長く一致するサーバー名を見つける (例: 'server' と 'server-ext' があれば後者を優先)
                if found_server_key is None or len(server_key) > len(found_server_key):
                    found_server_key = server_key

        if found_server_key:
            server_name = self._server_definitions[found_server_key].name # 辞書に登録されている正式な名前
            original_tool_name = name_suffix[len(found_server_key) + 1:] # サーバー名 + ハイフンの後がツール名

        if not server_name or not original_tool_name:
            raise ValueError(f"No active MCP server found for tool name '{full_tool_name}'. Available servers: {list(self._server_definitions.keys())}")

        session = self._sessions.get(server_name)
        if not session:
            logger.error(f"MCP server '{server_name}' is not connected. Cannot execute tool '{original_tool_name}'.")
            raise ConnectionError(f"MCP Server '{server_name}' is not connected.")

        logger.info(f"MCP tool '{server_name}/{original_tool_name}' (Canonical: {full_tool_name}) executing with arguments {arguments}...")
        try:
            call_result = await asyncio.wait_for(session.call_tool(original_tool_name, arguments), timeout=180.0)

            if call_result is None:
                logger.error(f"Tool '{full_tool_name}' execution result was None.")
                raise RuntimeError(f"Tool execution failed for '{full_tool_name}': Received None result.")

            logger.info(f"Tool '{full_tool_name}' execution completed.")

            if call_result.isError:
                error_content = call_result.content if call_result.content else "Unknown tool error"
                error_text = " ".join([c.text for c in error_content if isinstance(c, types.TextContent)]) if error_content else "N/A"
                logger.error(f"Tool '{full_tool_name}' execution error in server: {error_text}")
                raise RuntimeError(f"Tool execution error in '{full_tool_name}': {error_text}")
            return call_result.content if call_result.content else []
        except asyncio.TimeoutError:
            logger.error(f"Tool '{full_tool_name}' execution timed out.")
            raise TimeoutError(f"Tool execution timed out for '{full_tool_name}'.")
        except ConnectionError as ce:
            logger.error(f"Tool '{full_tool_name}' execution connection error: {ce}")
            raise
        except Exception as e:
            logger.error(f"Tool '{full_tool_name}' execution unexpected error: {e}", exc_info=True)
            raise

    # --- リソース・プロンプト関連のメソッド（スタブ） ---
    async def get_available_resources(self) -> List[types.Resource]:
        """
        現在接続中の全ての有効なMCPサーバーから利用可能なリソースリストを取得する。
        返される Resource オブジェクトの URI には "{self._mcp_prefix}{server_name}:" プレフィックスが付与される。

        Returns:
            List[types.Resource]: 利用可能なリソースのリスト。URIにはプレフィックスが付与される。
        """
        if not self._is_initialized:
            logger.warning("MCPClientManager is not initialized. Returning empty resource list.")
            return []
        if self._is_shutting_down:
            logger.warning("MCPClientManager is shutting down. Returning empty resource list.")
            return []

        all_prefixed_resources: List[types.Resource] = []
        tasks = []
        async with self._lock:
            active_sessions_data = list(self._sessions.items())

        async def fetch_and_prefix_resources(name: str, session: ClientSession):
            capabilities = self._server_capabilities.get(name)
            # ケイパビリティでリソースサポートを確認
            if capabilities and capabilities.resources:
                try:
                    # タイムアウト付きでリソースリスト取得
                    # Pagination未対応: session.list_resources() の結果は最初のページのみ
                    list_result = await asyncio.wait_for(session.list_resources(), timeout=30.0)
                    if list_result and list_result.resources:
                        for resource in list_result.resources:
                            if resource:
                                # 新しい Resource オブジェクトを作成して URI を書き換える
                                try:
                                    prefixed_uri = f"{self._mcp_prefix}{name}:{resource.uri}"
                                    # 元のオブジェクトを変更せず、新しいオブジェクトを作成
                                    prefixed_resource = types.Resource(
                                        uri=prefixed_uri, # ここでプレフィックス付きURIを設定
                                        name=resource.name,
                                        description=resource.description,
                                        mimeType=resource.mimeType,
                                        size=resource.size,
                                        # 他の Resource フィールドがあればここに追加
                                    )
                                    all_prefixed_resources.append(prefixed_resource)
                                except Exception as prefix_err:
                                    logger.error(f"Server '{name}' resource '{resource.uri}' prefixing error: {prefix_err}", exc_info=False)
                            else:
                                logger.warning(f"Server '{name}' returned None resource object. Skipping.")
                    elif list_result is None:
                        logger.warning(f"Server '{name}' resource list retrieval result was None.")
                    # else: list_result.resources が空の場合は何もしない

                except asyncio.TimeoutError:
                    logger.error(f"Server '{name}' resource list retrieval timed out.")
                except McpError as e:
                    logger.error(f"Server '{name}' resource list retrieval error: {e.code}: {e.message}")
                except Exception as e:
                    logger.error(
                        f"Server '{name}' resource list retrieval or prefixing error: {e}", exc_info=False)
            else:
                logger.debug(f"Server '{name}' does not support resource functionality or does not have capability information.")

        for name, session in active_sessions_data:
            if session:
                tasks.append(fetch_and_prefix_resources(name, session))

        if tasks:
            await asyncio.gather(*tasks)

        logger.info(f"{len(all_prefixed_resources)} resource definitions obtained.")
        return all_prefixed_resources

    def _parse_mcp_uri(self, full_resource_uri: str) -> Tuple[str, str]:
        """
        プレフィックス付きURI ("{self._mcp_prefix}{server_name}:{original_uri}") からサーバー名と元のURIを抽出する。
        """
        if not full_resource_uri.startswith(self._mcp_prefix):
            raise ValueError(f"Invalid MCP resource URI. '{self._mcp_prefix}' prefix is required: {full_resource_uri}")

        # プレフィックスを除去
        uri_body = full_resource_uri[len(self._mcp_prefix):]

        # 最初の ":" でサーバー名と元のURIを分割
        parts = uri_body.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid MCP resource URI format. Server name and original URI must be separated by ':': {full_resource_uri}")

        server_name = parts[0]
        original_uri = parts[1]

        if not server_name:
            raise ValueError(f"Invalid MCP resource URI. Server name is empty: {full_resource_uri}")
        if not original_uri:
            raise ValueError(f"Invalid MCP resource URI. Original URI is empty: {full_resource_uri}")

        # サーバー名が有効かチェック
        if server_name not in self._server_definitions:
            raise ValueError(f"Server name '{server_name}' in URI is not configured: {full_resource_uri}")

        return server_name, original_uri

    async def read_mcp_resource(self, full_resource_uri: str) -> types.ReadResourceResult:
        """
        指定されたプレフィックス付きMCPリソースURIの内容を読み取る。

        Args:
            full_resource_uri: "{self._mcp_prefix}{server_name}:{original_uri}" 形式のURI。

        Returns:
            types.ReadResourceResult: サーバーから返されたリソースの読み取り結果。含まれる contents の URI は元の（プレフィックスなしの）URI。

        Raises:
            ValueError: URI形式が無効な場合、またはサーバー名が見つからない場合。
            ConnectionError: 対象サーバーが接続されていない場合。
            TimeoutError: サーバーからの応答がタイムアウトした場合。
            RuntimeError: サーバー側でエラーが発生した場合 (McpError含む)。
            RuntimeError: MCPClientManagerが初期化されていないかシャットダウン中の場合。
        """
        if not self._is_initialized:
            raise RuntimeError("MCPClientManager is not initialized.")
        if self._is_shutting_down:
            raise RuntimeError("MCPClientManager is shutting down. Cannot read resources.")

        logger.info(f"Attempting to read MCP resource '{full_resource_uri}'...")
        try:
            server_name, original_uri = self._parse_mcp_uri(full_resource_uri)
        except ValueError as e:
            logger.error(f"Resource URI parsing failed: {e}")
            raise # エラーをそのまま再raise

        session = self._sessions.get(server_name)
        if not session:
            logger.error(f"MCP server '{server_name}' is not connected. Cannot read resource '{original_uri}'.")
            raise ConnectionError(f"MCP Server '{server_name}' is not connected.")

        logger.debug(f"Using server '{server_name}' to read original URI '{original_uri}'...")
        try:
            # 読み取りリクエストにタイムアウトを追加 (例: 30秒)
            read_result = await asyncio.wait_for(session.read_resource(original_uri), timeout=30.0)

            if read_result is None:
                logger.error(f"Resource '{full_resource_uri}' read result was None.")
                # リソースが存在しない場合は McpError が送出されるはずだが、念のため
                raise RuntimeError(f"Failed to read resource '{full_resource_uri}': Received None result.")

            logger.info(f"Resource '{full_resource_uri}' read completed.")
            # 結果の contents[].uri は元のURIのままなので、書き換えは不要
            return read_result

        except asyncio.TimeoutError:
            logger.error(f"Resource '{full_resource_uri}' read timed out.")
            raise TimeoutError(f"Reading resource '{full_resource_uri}' timed out.")
        except McpError as e:
            # サーバーが返すエラー (例: Resource Not Found -32002)
            logger.error(f"Server error reading resource '{full_resource_uri}' (Code: {e.code}): {e.message}")
            # エラー内容に応じて特定の例外に変換するか、RuntimeErrorでラップする
            # ここでは RuntimeError でラップする
            raise RuntimeError(f"Server error reading resource '{full_resource_uri}' (Code: {e.code}): {e.message}") from e
        except ConnectionError as ce: # セッションが途中で切れた場合など
            logger.error(f"Resource '{full_resource_uri}' read error: {ce}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading resource '{full_resource_uri}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error reading resource '{full_resource_uri}'.") from e

    async def get_available_prompts(self) -> Dict[str, List[Dict[str, Any]]]:
        logger.warning("get_available_prompts is not implemented yet.")
        return {}

    async def get_mcp_prompt(self, full_prompt_name: str, arguments: Dict[str, Any]) -> Any:
        logger.warning("get_mcp_prompt is not implemented yet.")
        raise NotImplementedError

    # --- クリーンアップ ---
    async def shutdown(self):
        """
        全てのMCPサーバー接続と関連プロセスを安全に閉じる。
        アプリケーション終了時に呼び出す。
        """
        async with self._lock:
            
            if self._is_shutting_down:
                logger.info("Shutdown process is already in progress.")
                return
            if not self._is_initialized and not self._connection_tasks:
                logger.info("MCPClientManager is already shut down or not initialized.")
                return

            self._is_shutting_down = True  # シャットダウン開始フラグを設定
            logger.info("Starting shutdown of MCP connections and processes...")
            self._is_initialized = False  # 新規初期化をブロック
            

            tasks_to_cancel = list(self._connection_tasks.values())
            self._connection_tasks.clear()  # 早めにクリアして新規タスク追加を防ぐ

            # 初期接続Futureもキャンセル状態にする
            futures_to_cancel = list(self._initial_connection_futures.values())
            self._initial_connection_futures.clear()
            for fut in futures_to_cancel:
                if not fut.done():
                    fut.cancel()

            if tasks_to_cancel:
                for task in tasks_to_cancel:
                    if not task.done():
                        task.cancel()
                results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                cancelled_count = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
                error_count = sum(1 for r in results if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError))
                logger.info(f"{len(tasks_to_cancel)} connections/monitoring tasks completed. (Cancelled: {cancelled_count}, Errors: {error_count})")
            else:
                logger.info("No tasks to cancel.")

            # セッションとケイパビリティ情報をクリア (既に接続タスクのfinallyで削除されているはずだが念のため)
            self._sessions.clear()
            self._server_capabilities.clear()
            self._server_definitions.clear()  # 定義もクリア
            await asyncio.sleep(0.5)

            logger.info("MCP connections and processes shutdown complete.")            
            self._is_shutting_down = False # 必要であれば再度初期化可能にする場合