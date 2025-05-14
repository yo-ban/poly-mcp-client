import asyncio
import json
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional, Literal, Tuple, Set
import logging
from pydantic import ValidationError
import time

# MCP SDKのインポート
from mcp import ClientSession, StdioServerParameters, McpError
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
import mcp.types as types
import httpx

from .constants import MCP_PREFIX
from .models import (
    StdioServerConfig, 
    HttpServerConfig, 
    StreamableHttpServerConfig,
    McpServersConfig, 
    InternalServerDefinition, 
    CanonicalToolParameter, 
    CanonicalToolDefinition,
    ServerConfig 
)

# --- Keep-Alive設定 ---
PING_INTERVAL = 30.0  # Pingを送信する間隔（秒）
PING_TIMEOUT = 10.0   # Ping応答のタイムアウト（秒）

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

    def _parse_and_validate_config(
        self,
        config_data: Optional[Dict[str, Any]],
        context: str = "operation" # "initialization" or "update"
    ) -> Optional[Dict[str, InternalServerDefinition]]:
        """設定データをパース・バリデーションし、内部定義の辞書を返す。エラー時は None を返す。"""
        servers_dict_data: Optional[Dict] = None
        if config_data and isinstance(config_data, dict) and "mcpServers" in config_data and isinstance(config_data.get("mcpServers"), dict):
            servers_dict_data = config_data["mcpServers"]
        elif config_data: # config_data はあるが形式が不正
            logger.error(f"Invalid configuration data for {context}. Top level must contain 'mcpServers' object.")
            return None
        # config_data が None の場合は空の辞書を処理対象とする

        new_definitions: Dict[str, InternalServerDefinition] = {}
        try:
            # servers_dict_data が None の場合は空の辞書を渡す
            parsed_servers = McpServersConfig.model_validate(servers_dict_data or {})
            for server_name, server_config in parsed_servers.root.items():
                internal_def = InternalServerDefinition(
                    name=server_name,
                    type=server_config.type,
                    config=server_config
                )
                new_definitions[server_name] = internal_def
            logger.info(f"{len(new_definitions)} MCP server configurations validated successfully for {context}.")
            return new_definitions
        except ValidationError as e:
            logger.error(f"MCP server configuration validation error during {context}:\n{e}")
            return None # バリデーションエラー時は None
        except Exception as e:
            logger.error(f"Unexpected error during configuration parsing for {context}: {e}", exc_info=True)
            return None # その他のエラー時も None

    async def _reconcile_connections(
        self,
        new_definitions: Dict[str, InternalServerDefinition],
        is_initialization: bool = False
    ):
        """現在の接続状態を新しい定義に合わせて調整（開始/停止/再接続）する。"""
        current_server_names = set(self._server_definitions.keys())
        new_server_names = set(new_definitions.keys())

        servers_to_remove = current_server_names - new_server_names
        servers_to_add = new_server_names - current_server_names
        servers_to_check_update = current_server_names.intersection(new_server_names)

        servers_to_update: Set[str] = set()
        servers_needing_restart: Set[str] = set() # 接続が切れているが設定は残るサーバー

        for name in servers_to_check_update:
            config_changed = False
            # is_initialization が True の場合、既存の定義はないので比較しない
            if not is_initialization and name in self._server_definitions and self._server_definitions[name].config != new_definitions[name].config:
                servers_to_update.add(name)
                config_changed = True

            # 設定変更がないサーバーで、接続がアクティブでないものを再起動対象とする
            if not config_changed:
                # セッションがないか、接続タスクが完了している場合 (接続が切れているとみなす)
                task = self._connection_tasks.get(name)
                if name not in self._sessions or (task and task.done()):
                    logger.info(f"Server '{name}' config unchanged but connection is inactive. Marking for restart.")
                    servers_needing_restart.add(name)


        context = "initialization" if is_initialization else "update"
        logger.info(f"Connection Reconciliation ({context}) - Remove: {list(servers_to_remove)}, Add: {list(servers_to_add)}, Update: {list(servers_to_update)}, Restart (inactive): {list(servers_needing_restart)}")

        # --- 既存接続の停止 (削除・更新対象) ---
        # Note: 再起動対象(servers_needing_restart)は、既に接続が切れているので停止処理は不要
        servers_to_stop = servers_to_remove.union(servers_to_update)
        if servers_to_stop:
            logger.info(f"Stopping connections for servers: {list(servers_to_stop)}")
            stop_tasks = [self._stop_connection(name, reason=f"configuration {context}") for name in servers_to_stop]
            await asyncio.gather(*stop_tasks, return_exceptions=True) # エラーは無視しないが続行
            logger.info(f"Finished stopping connections for {len(servers_to_stop)} servers.")
        else:
            logger.info("No servers need to be stopped.")

        # --- 新規接続の開始 (追加・更新・再起動対象) ---
        servers_to_start = servers_to_add.union(servers_to_update).union(servers_needing_restart)
        if servers_to_start:
            logger.info(f"Starting/Restarting connections for servers: {list(servers_to_start)}")
            for name in servers_to_start:
                definition = new_definitions.get(name) # Use get() for safety
                if not definition:
                    logger.warning(f"Definition for server '{name}' not found in new_definitions during start phase. Skipping.")
                    continue

                # 新しい接続 or 更新 or 再起動なので、新しい Future を作成または取得
                # 完了していないFutureを再利用しないように、完了済みの場合は常に新しいものを作成する
                if name not in self._initial_connection_futures or self._initial_connection_futures[name].done():
                    self._initial_connection_futures[name] = asyncio.Future()

                future = self._initial_connection_futures[name]
                self._server_definitions[name] = definition # 内部定義を更新/追加
                # _start_connection 内で既存タスクのチェックは行われる
                self._start_connection(name, definition, future) # タスク開始

            logger.info(f"Initiated starting/restarting connections for {len(servers_to_start)} servers.")
        else:
            logger.info("No new servers need to be started or restarted.")

        # --- 内部定義の最終調整 (削除されたサーバー定義を確実に消す) ---
        # 停止されたサーバーの定義は _stop_connection で削除される
        # ここでは、新しい定義セットに基づいて最終的な状態を確定する
        # （servers_to_startで追加/更新された定義は既に反映されているはず）
        current_defs = set(self._server_definitions.keys())
        defs_to_remove_finally = current_defs - new_server_names
        for name_to_remove in defs_to_remove_finally:
            self._server_definitions.pop(name_to_remove, None)
            self._initial_connection_futures.pop(name_to_remove, None) # 関連Futureも削除


        logger.info(f"Connection reconciliation complete ({context}).")
        logger.info(f"Active server definitions: {list(self._server_definitions.keys())}")


    async def initialize(self, config_path: Optional[str] = None, config_data: Optional[Dict[str, Any]] = None):
        """指定された設定ファイルまたはデータからMCPサーバー接続を初期化する。"""
        async with self._lock:
            if self._is_initialized:
                logger.warning("MCPClientManager is already initialized.")
                return
            if self._is_shutting_down:
                logger.warning("MCPClientManager is shutting down. Cannot initialize.")
                return

            logger.info("MCPClientManager is initializing...")

            # --- 設定の読み込み ---
            raw_config_data_dict: Optional[Dict[str, Any]] = None # config_data を受け取る変数
            if config_path:
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        raw_config_data_dict = json.load(f)
                except FileNotFoundError:
                    logger.error(f"Configuration file not found: {config_path}")
                    return
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in configuration file: {e}")
                    return
            elif config_data:
                raw_config_data_dict = config_data
            else:
                logger.warning("No configuration file path or data provided. Skipping initialization.")
                # 空でも初期化は完了とする
                self._is_initialized = True
                return

            # --- 設定のパースとバリデーション ---
            new_definitions = self._parse_and_validate_config(raw_config_data_dict, context="initialization")

            if new_definitions is None: # パース/バリデーションエラー
                # ログは _parse_and_validate_config 内で出力済み
                return

            # --- 接続前の状態クリア ---
            # initialize では既存の接続はないはずだが、念のためクリア
            # (update と異なり、既存タスク停止は不要)
            self._server_definitions.clear()
            self._connection_tasks.clear() # 既存タスクは考慮しない
            self._sessions.clear()
            self._server_capabilities.clear()
            self._initial_connection_futures.clear() # Future もクリア

            # --- 新しい Future を準備 ---
            for name in new_definitions.keys():
                self._initial_connection_futures[name] = asyncio.Future()

            # --- 接続調整 (実質、全サーバー追加) ---
            await self._reconcile_connections(new_definitions, is_initialization=True)

            self._is_initialized = True
            logger.info("MCPClientManager initialization complete. Connections will be attempted in the background.")
            logger.info("To wait for initial connections, call await manager.wait_for_initial_connections()")


    async def wait_for_connections(self, timeout: Optional[float] = None) -> Dict[str, Tuple[bool, Optional[Exception]]]:
        """
        initializeまたはupdate_configurationで開始/再開されたサーバーへの
        接続試行が完了するまで待機する。
        (コメントを修正、ロジック変更なし)
        """
        if not self._is_initialized and not self._server_definitions: # server_definitions もチェック
            logger.warning("Manager is not initialized or no servers are configured.")
            return {}

        # 完了していない Future のみを待機対象とする方が効率的かもしれないが、
        # 現状は辞書にあるもの全てを待つ
        futures_to_wait = [
            f for f in self._initial_connection_futures.values() if not f.done()
        ]
        # futures = list(self._initial_connection_futures.values()) # 以前の実装
        if not futures_to_wait:
            logger.info("No pending initial connection futures to wait for.")
            # 既に完了しているFutureの結果も含めて返す (オプション)
            results: Dict[str, Tuple[bool, Optional[Exception]]] = {}
            for name, fut in self._initial_connection_futures.items():
                try:
                    result = fut.result() # Raises exception if future has error
                    results[name] = (True, None)
                except asyncio.CancelledError:
                    results[name] = (False, asyncio.CancelledError(f"Connection for '{name}' was cancelled."))
                except Exception as e:
                    results[name] = (False, e)
            return results
            # return {} # 待機対象がなければ空を返すシンプルな実装

        logger.info(f"{len(futures_to_wait)} servers to wait for connection (timeout: {timeout}s)...")

        done, pending = await asyncio.wait(futures_to_wait, timeout=timeout, return_when=asyncio.ALL_COMPLETED)

        results: Dict[str, Tuple[bool, Optional[Exception]]] = {}
        server_name_map = {f: name for name, f in self._initial_connection_futures.items()}

        # Fill results from completed futures (done set)
        for fut in done:
            server_name = server_name_map.get(fut, "Unknown")
            try:
                result = fut.result()
                results[server_name] = (True, None)
                logger.info(f"Server '{server_name}' connection successful or already completed.")
            except asyncio.CancelledError:
                results[server_name] = (False, asyncio.CancelledError(f"Initial connection for '{server_name}' was cancelled."))
                logger.warning(f"Server '{server_name}' initial connection cancelled.")
            except Exception as e:
                results[server_name] = (False, e)
                logger.error(f"Server '{server_name}' initial connection error: {e}", exc_info=False)

        # Handle pending futures (timeout)
        if pending:
            logger.warning(f"Timeout ({timeout}s) reached. Some connections did not complete: {len(pending)}")
            for fut in pending:
                server_name = server_name_map.get(fut, "Unknown")
                # Do not cancel the future here, let the connection attempt continue
                # fut.cancel() # 削除
                results[server_name] = (False, asyncio.TimeoutError(f"Initial connection timed out after {timeout}s"))

        # Include results for futures that might have completed before the wait started
        for name, fut in self._initial_connection_futures.items():
            if name not in results and fut.done(): # まだ結果に含まれていない完了済みFuture
                try:
                    result = fut.result()
                    results[name] = (True, None)
                except asyncio.CancelledError:
                    results[name] = (False, asyncio.CancelledError(f"Connection for '{name}' was cancelled."))
                except Exception as e:
                    results[name] = (False, e)


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

    async def _stop_connection(self, server_name: str, reason: str = "stopping connection"):
        """指定されたサーバーの接続監視タスクを停止し、関連データをクリーンアップする"""
        logger.info(f"Stopping connection for server '{server_name}' ({reason})...")

        # 1. タスクを取得してキャンセル
        task = self._connection_tasks.pop(server_name, None)
        if task and not task.done():
            logger.debug(f"Cancelling connection task for server '{server_name}'...")
            task.cancel()
            try:
                # タスクの完了 (キャンセル含む) を待つ (タイムアウトを設定)
                await asyncio.wait_for(task, timeout=10.0)
                logger.info(f"Connection task for server '{server_name}' cancelled successfully.")
            except asyncio.CancelledError:
                logger.info(f"Connection task for server '{server_name}' cancellation confirmed.")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for connection task cancellation of server '{server_name}'.")
            except Exception as e:
                logger.error(f"Error during connection task cancellation for server '{server_name}': {e}", exc_info=False)
        elif task and task.done():
            logger.debug(f"Connection task for server '{server_name}' was already done.")
        else:
            logger.debug(f"No active connection task found for server '{server_name}'.")


        # 2. 内部状態から削除 (タスクの finally ブロックでも削除されるが、念のため)
        #    ロックの外で実行される想定なので、pop を使う
        self._sessions.pop(server_name, None)
        self._server_capabilities.pop(server_name, None)
        definition = self._server_definitions.pop(server_name, None) # 定義も削除
        future = self._initial_connection_futures.pop(server_name, None)

        # 3. 初期接続 Future があればキャンセル (完了していなければ)
        if future and not future.done():
            future.cancel()

        if definition:
            logger.info(f"Stopped and cleaned up server '{server_name}' ({definition.type}).")
        else:
            logger.info(f"Cleaned up internal state for server '{server_name}' (definition not found).")


    async def update_configuration(self, config_path: Optional[str] = None, config_data: Optional[Dict[str, Any]] = None):
        """現在のMCPサーバー接続設定を、指定された設定ファイルまたはデータから更新する。"""
        async with self._lock:
            if self._is_shutting_down:
                logger.warning("Manager is shutting down. Cannot update configuration.")
                return
            if not self._is_initialized:
                logger.warning("Manager is not initialized. Cannot update configuration. Call initialize first.")
                return

            # --- 設定の読み込み ---
            raw_config_data_dict: Optional[Dict[str, Any]] = None # config_data を受け取る変数
            if config_path:
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        raw_config_data_dict = json.load(f)
                except FileNotFoundError:
                    logger.error(f"Configuration file not found: {config_path}")
                    return
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in configuration file: {e}")
                    return
            elif config_data:
                raw_config_data_dict = config_data
            else:
                logger.warning("No configuration file path or data provided. Using empty configuration.")
                raw_config_data_dict = {}


            logger.info("Updating MCP server configuration...")

            # --- 設定のパースとバリデーション ---
            new_definitions = self._parse_and_validate_config(raw_config_data_dict, context="update")

            if new_definitions is None: # パース/バリデーションエラー
                return

            # --- 接続調整 ---
            await self._reconcile_connections(new_definitions, is_initialization=False)


    async def _connect_and_monitor(self, name: str, definition: InternalServerDefinition, initial_conn_future: asyncio.Future):
        """サーバーへの接続、セッション確立、プロセス監視を行うコルーチン (Futureを引数に追加)"""
        retry_delay = 5
        max_retry_delay = 60
        initial_attempt = True # 最初の接続試行かどうかを追跡
        exit_stack = AsyncExitStack()

        while not self._is_shutting_down:  # 初期化フラグではなくシャットダウンフラグを見る
            session = None
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
                        logger.error(f"Invalid config type for http server: {name}")
                        connection_error = TypeError("Invalid config type for http server.")
                        break
                    http_config = definition.config
                    server_url = http_config.url
                    logger.info(f"Attempting connection to server '{name}' (http): {server_url}")

                    connection_established_in_this_loop = False
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

                        connection_established_in_this_loop = True
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

                    # 接続成功 -> Pingループ開始
                    if connection_established_in_this_loop and session:
                        # 初期接続Future設定 (元のコードと同じ)
                        if initial_attempt and not initial_conn_future.done():
                            initial_conn_future.set_result(True)
                        initial_attempt = False

                        # --- Keep-Alive Loop (HTTP/SSE with Ping) ---
                        logger.info(f"Starting keep-alive ping loop for server '{name}' (http). Interval: {PING_INTERVAL}s")
                        last_ping_time = time.monotonic() # Keep track of last successful ping or connection time

                        while not self._is_shutting_down and name in self._sessions:
                            current_session = self._sessions.get(name)
                            if not current_session:
                                logger.warning(f"Session object for '{name}' missing during ping loop.")
                                connection_error = RuntimeError("Session object missing")
                                break # Exit inner loop

                            try:
                                # Wait until next ping time or shutdown/removal
                                time_since_last_ping = time.monotonic() - last_ping_time
                                wait_time = max(0, PING_INTERVAL - time_since_last_ping)
                                logger.debug(f"[{name}] Waiting {wait_time:.1f}s for next ping...")

                                # Sleep for the calculated wait time, but check for shutdown periodically
                                sleep_task = asyncio.create_task(asyncio.sleep(wait_time))
                                done, pending = await asyncio.wait({sleep_task}, return_when=asyncio.FIRST_COMPLETED)

                                # Check if shutdown/removal happened during sleep
                                if self._is_shutting_down:
                                    logger.info(f"Shutdown requested during ping interval for server '{name}'.")
                                    sleep_task.cancel() # Cancel the sleep task if still pending
                                    break # Exit inner loop
                                if name not in self._sessions:
                                    logger.warning(f"Server '{name}' removed during ping interval.")
                                    sleep_task.cancel()
                                    break # Exit inner loop

                                # --- Send Ping ---
                                logger.debug(f"Sending MCP ping to server '{name}' (http).")
                                await asyncio.wait_for(current_session.ping(), timeout=PING_TIMEOUT)
                                last_ping_time = time.monotonic() # Update time after successful ping
                                logger.debug(f"MCP ping to server '{name}' (http) successful.")

                            except asyncio.TimeoutError:
                                logger.warning(f"MCP ping to server '{name}' (http) timed out after {PING_TIMEOUT}s. Assuming connection lost.")
                                connection_error = asyncio.TimeoutError("Ping timeout")
                                break # Exit inner loop to trigger reconnection
                            except (McpError, ConnectionError, AttributeError, httpx.TransportError) as e: # Added httpx.TransportError
                                logger.error(f"Error during MCP ping to server '{name}' (http): {type(e).__name__}: {e}. Assuming connection lost.", exc_info=False)
                                connection_error = e # Record the error
                                break # Exit inner loop to trigger reconnection
                            except asyncio.CancelledError:
                                logger.info(f"Ping loop cancelled for server '{name}'.")
                                connection_error = asyncio.CancelledError("Ping loop cancelled")
                                break # Exit inner loop and outer loop
                            except Exception as e: # Catch unexpected errors during ping loop
                                logger.error(f"Unexpected error in ping loop for server '{name}': {e}", exc_info=True)
                                connection_error = e
                                break # Exit inner loop

                        # --- End of Keep-Alive Loop ---
                        if self._is_shutting_down:
                            logger.info(f"Shutdown requested. Stopping monitoring of server '{name}' (http).")
                            break # Exit outer loop
                        elif name not in self._sessions:
                            logger.info(f"Server '{name}' was removed during ping loop. Stopping monitoring task.")
                            break # Exit outer loop
                        else:
                            # Inner loop broken due to error (ping failure/timeout/other exception)
                            logger.warning(f"Keep-alive loop ended for server '{name}' (http). Reason: {type(connection_error).__name__ if connection_error else 'Unknown'}. Attempting reconnect.")
                            # Let the outer loop handle retry logic

                    else:
                        # 接続に失敗した場合 (connection_error が設定されているはず)
                        # ループの最後でリトライされる
                        pass
                elif definition.type == "streamable-http":
                    if not isinstance(definition.config, StreamableHttpServerConfig):
                        logger.error(f"Invalid config type for streamable-http server: {name}")
                        connection_error = TypeError("Invalid config type for streamable-http server.")
                        break
                    streamable_http_config = definition.config
                    server_url = streamable_http_config.url
                    logger.info(f"Attempting connection to server '{name}' (streamable-http): {server_url}")

                    connection_established_in_this_loop = False
                    try:
                        transport_streams = await exit_stack.enter_async_context(streamablehttp_client(server_url))
                        reader, writer, _ = transport_streams
                        logger.info(f"Transport connection to server '{name}' (streamable-http) complete.")

                        session = await exit_stack.enter_async_context(ClientSession(reader, writer))
                        init_result = await asyncio.wait_for(session.initialize(), timeout=30.0)

                        self._sessions[name] = session
                        capabilities = init_result.capabilities
                        self._server_capabilities[name] = capabilities
                        logger.info(f"MCP connection to server '{name}' (streamable-http) established.")
                        if capabilities:
                            logger.info(f"Server '{name}' (streamable-http) capabilities: {capabilities}")
                        
                        connection_established_in_this_loop = True
                        retry_delay = 5

                    except httpx.RequestError as e:
                        logger.error(f"HTTP request error during connection to server '{name}' (streamable-http): {e}")
                        connection_error = e
                    except httpx.HTTPStatusError as e:
                        logger.error(f"Server '{name}' (streamable-http) returned error status: {e.response.status_code} {e.response.reason_phrase}")
                        connection_error = e
                    except ConnectionRefusedError as e:
                        logger.error(f"Connection to server '{name}' (streamable-http) refused. Check URL: {server_url}")
                        connection_error = e
                    except asyncio.TimeoutError as e: # initialize のタイムアウト
                        logger.error(f"Server '{name}' (streamable-http) initialization timed out.")
                        connection_error = e
                    except Exception as e:
                        logger.error(f"Unexpected connection error during connection to server '{name}' (streamable-http): {e}", exc_info=False)
                        connection_error = e
                        break 

                    if connection_established_in_this_loop and session:
                        if initial_attempt and not initial_conn_future.done():
                            initial_conn_future.set_result(True)
                        initial_attempt = False

                        logger.info(f"Starting keep-alive ping loop for server '{name}' (streamable-http). Interval: {PING_INTERVAL}s")
                        last_ping_time = time.monotonic()

                        while not self._is_shutting_down and name in self._sessions:
                            current_session = self._sessions.get(name)
                            if not current_session:
                                logger.warning(f"Session object for '{name}' missing during ping loop.")
                                connection_error = RuntimeError("Session object missing")
                                break

                            try:
                                time_since_last_ping = time.monotonic() - last_ping_time
                                wait_time = max(0, PING_INTERVAL - time_since_last_ping)
                                logger.debug(f"[{name}] Waiting {wait_time:.1f}s for next ping...")

                                sleep_task = asyncio.create_task(asyncio.sleep(wait_time))
                                done, pending = await asyncio.wait({sleep_task}, return_when=asyncio.FIRST_COMPLETED)

                                if self._is_shutting_down:
                                    logger.info(f"Shutdown requested during ping interval for server '{name}'.")
                                    sleep_task.cancel()
                                    break
                                if name not in self._sessions:
                                    logger.warning(f"Server '{name}' removed during ping interval.")
                                    sleep_task.cancel()
                                    break

                                logger.debug(f"Sending MCP ping to server '{name}' (streamable-http).")
                                await asyncio.wait_for(current_session.ping(), timeout=PING_TIMEOUT)
                                last_ping_time = time.monotonic()
                                logger.debug(f"MCP ping to server '{name}' (streamable-http) successful.")

                            except asyncio.TimeoutError:
                                logger.warning(f"MCP ping to server '{name}' (streamable-http) timed out after {PING_TIMEOUT}s. Assuming connection lost.")
                                connection_error = asyncio.TimeoutError("Ping timeout")
                                break
                            except (McpError, ConnectionError, AttributeError, httpx.TransportError) as e:
                                logger.error(f"Error during MCP ping to server '{name}' (streamable-http): {type(e).__name__}: {e}. Assuming connection lost.", exc_info=False)
                                connection_error = e
                                break
                            except asyncio.CancelledError:
                                logger.info(f"Ping loop cancelled for server '{name}'.")
                                connection_error = asyncio.CancelledError("Ping loop cancelled")
                                break 
                            except Exception as e: 
                                logger.error(f"Unexpected error in ping loop for server '{name}': {e}", exc_info=True)
                                connection_error = e
                                break
                        
                        if self._is_shutting_down:
                            logger.info(f"Shutdown requested. Stopping monitoring of server '{name}' (streamable-http).")
                            break
                        elif name not in self._sessions:
                            logger.info(f"Server '{name}' was removed during ping loop. Stopping monitoring task.")
                            break
                        else:
                            logger.warning(f"Keep-alive loop ended for server '{name}' (streamable-http). Reason: {type(connection_error).__name__ if connection_error else 'Unknown'}. Attempting reconnect.")
                    else:
                        pass # Connection failed, retry handled by outer loop
                else:
                    connection_error = ValueError(f"Unknown server type: {definition.type} ({name})")
                    logger.error(connection_error)
                    break

            # --- 接続エラー/ループ終了時の共通処理 ---
            except (ConnectionRefusedError, FileNotFoundError, asyncio.TimeoutError) as e:
                connection_error = e # エラーを記録
                logger.error(f"Failed to connect to server '{name}': {e}", exc_info=False)
                # initial_attempt フラグは finally で False にする
            except asyncio.CancelledError:
                logger.info(f"Connection task for server '{name}' was cancelled.")
                connection_error = asyncio.CancelledError("Task cancelled")  # キャンセルも記録
                break # キャンセルされたらループ終了
            except Exception as e:
                connection_error = e # その他の予期せぬエラー
                logger.error(f"Unexpected error occurred during connection or monitoring of server '{name}': {e}", exc_info=False)
                # initial_attempt フラグは finally で False にする
            finally:
                # --- リソースクリーンアップ & Future処理 ---
                if initial_attempt and not initial_conn_future.done():
                    error_to_set = connection_error or RuntimeError(f"Initial connection attempt failed for '{name}'.")
                    if isinstance(error_to_set, asyncio.CancelledError):
                        initial_conn_future.cancel()
                        logger.info(f"Initial connection future cancelled for '{name}'.")
                    else:
                        initial_conn_future.set_exception(error_to_set)
                        logger.warning(f"Initial connection future set to failed for '{name}': {error_to_set}")
                initial_attempt = False

                # セッション/ケイパビリティ情報のクリーンアップ
                self._sessions.pop(name, None)
                self._server_capabilities.pop(name, None)
                logger.debug(f"Cleaned session/capability state for server '{name}' after attempt/disconnection.")

                # ExitStackのクリーンアップ
                try:
                    await exit_stack.aclose()
                    exit_stack = AsyncExitStack() # 次の試行のために再初期化
                    logger.debug(f"Cleaned up resource stack for server '{name}'.")
                except Exception as e_stack:
                    logger.error(f"Error cleaning up resource stack for server '{name}': {e_stack}")

                if self._is_shutting_down:
                    logger.info(f"Shutdown in progress. Reconnection for server '{name}' will not be attempted.")
                    break

            # 再接続ロジック (キャンセルされていない場合)
            # リトライすべきでないエラータイプの場合はループを抜ける
            if isinstance(connection_error, (TypeError, ValueError, NotImplementedError)):
                logger.error(f"Unrecoverable error for server '{name}' ({definition.type}): {connection_error}. Stopping connection attempts.")
                break
            if isinstance(connection_error, asyncio.CancelledError):
                break # キャンセルならリトライしない

            logger.info(f"Attempting reconnection to server '{name}' in {retry_delay} seconds...")
            try:
                
                await asyncio.sleep(retry_delay)  # キャンセル可能にする
            except asyncio.CancelledError:
                logger.info(f"Reconnection wait cancelled ({name}).")
                break  # キャンセルされたらループ終了
            retry_delay = min(retry_delay * 2, max_retry_delay)

        logger.info(f"Connection and monitoring task for server '{name}' ({definition.type}) ended.")

        try:
            await exit_stack.aclose()
            logger.debug(f"Cleaned up resource stack for server '{name}' at task end.")
        except Exception as e_stack:
            logger.error(f"Error cleaning up resource stack for server '{name}' at task end: {e_stack}")

        # タスク終了時に Future がまだ完了していなければ、エラー状態にする
        if not initial_conn_future.done():
            final_error = connection_error or RuntimeError(f"Connection task for '{name}' ended unexpectedly.")
            if isinstance(final_error, asyncio.CancelledError):
                initial_conn_future.cancel()
            else:
                initial_conn_future.set_exception(final_error)
            logger.warning(f"Connection future for '{name}' set to failure/cancelled as task ended.")

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
            
            # _stop_connection を使って各タスクを停止
            tasks_to_stop = list(self._connection_tasks.keys()) # コピーを作成
            stop_tasks = []
            if tasks_to_stop:
                logger.info(f"Stopping {len(tasks_to_stop)} active connections...")
                for name in tasks_to_stop:
                    stop_tasks.append(self._stop_connection(name, reason="shutdown"))

                results = await asyncio.gather(*stop_tasks, return_exceptions=True)
                stopped_count = len(results)
                error_count = sum(1 for r in results if isinstance(r, Exception))
                logger.info(f"{stopped_count} connection stop routines completed. (Errors: {error_count})")
            else:
                logger.info("No active connections to stop.")

            # _stop_connection 内でクリアされるが念のため
            self._connection_tasks.clear()
            self._sessions.clear()
            self._server_capabilities.clear()
            self._server_definitions.clear()
            self._initial_connection_futures.clear()
            await asyncio.sleep(0.5)

            logger.info("MCP connections and processes shutdown complete.")            
            self._is_shutting_down = False # 必要であれば再度初期化可能にする場合